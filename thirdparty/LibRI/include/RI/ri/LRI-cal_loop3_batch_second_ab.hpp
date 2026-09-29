// Batched second-operand version of the four GW/EXX loop3 contractions.
#pragma once

#include "LRI.h"
#include "LRI_Cal_Aux.h"
#include "../global/Array_Operator.h"
#include "../global/Tensor_Multiply.h"

#include <algorithm>
#include <omp.h>
#include <stdexcept>
#ifdef __MKL_RI
#include <mkl_service.h>
#endif

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::cal_loop3_batch_second_ab(
	const std::vector<Label::ab_ab> &labels,
	const std::map<std::string,std::string> &second_ab_names,
	std::map<std::string,Tensor_map> &results_out,
	const double fac_add_Ds)
{
	for(const Label::ab_ab &label : labels)
		switch(label)
		{
			case Label::ab_ab::a0b0_a1b1:
			case Label::ab_ab::a0b0_a1b2:
			case Label::ab_ab::a0b0_a2b1:
			case Label::ab_ab::a0b0_a2b2:
				break;
			default:
				throw std::invalid_argument("cal_loop3_batch_second_ab: label "
					+ Label_Tools::get_name(label) + " is not implemented");
		}
	if(labels.empty() || second_ab_names.empty())
	{
		results_out.clear();
		return;
	}

	std::vector<std::string> result_names, dataset_names;
	result_names.reserve(second_ab_names.size());
	dataset_names.reserve(second_ab_names.size());
	for(const auto &name : second_ab_names)
	{
		result_names.push_back(name.first);
		dataset_names.push_back(name.second);
	}
	std::vector<Tensor_map> Ds_results(dataset_names.size());

	using namespace Array_Operator;
	using Ds_map = std::map<TA, std::map<TAC, Tensor<Tdata>>>;
	const auto find_data_name = [this](const Label::ab label) -> const std::string &
	{
		const auto it = this->data_ab_name.find(label);
		if(it == this->data_ab_name.end() || it->second.empty())
			throw std::invalid_argument("cal_loop3_batch_second_ab: missing shared operand "
				+ Label_Tools::get_name(label));
		return it->second;
	};
	const std::string name_a = find_data_name(Label::ab::a);
	const std::string name_b = find_data_name(Label::ab::b);
	const std::string name_a0b0 = find_data_name(Label::ab::a0b0);
	const auto second_label = [](const Label::ab_ab label)
	{
		switch(label)
		{
			case Label::ab_ab::a0b0_a1b1: return Label::ab::a1b1;
			case Label::ab_ab::a0b0_a1b2: return Label::ab::a1b2;
			case Label::ab_ab::a0b0_a2b1: return Label::ab::a2b1;
			case Label::ab_ab::a0b0_a2b2: return Label::ab::a2b2;
			default: throw std::invalid_argument("cal_loop3_batch_second_ab: unsupported label");
		}
	};
	const auto find_pack = [this](const std::string &name) -> const Data_Pack<TA,TC,Tdata> &
	{
		const auto it = this->data_pool.find(name);
		if(it == this->data_pool.end())
			throw std::invalid_argument("cal_loop3_batch_second_ab: unknown dataset " + name);
		return it->second;
	};
	const auto &pack_a = find_pack(name_a);
	const auto &pack_b = find_pack(name_b);

	std::vector<LRI_Cal_Tools<TA,TC,Tdata>> tools;
	tools.reserve(second_ab_names.size());
	for(const std::string &name : dataset_names)
	{
		find_pack(name);
		std::unordered_map<Label::ab,std::string> names;
		names.reserve(7);
		names.emplace(Label::ab::a, name_a);
		names.emplace(Label::ab::b, name_b);
		names.emplace(Label::ab::a0b0, name_a0b0);
		for(const Label::ab_ab &label : labels)
			names.emplace(second_label(label), name);
		tools.emplace_back(this->period, this->data_pool, names);
	}

	const bool need_a_transpose = std::find(labels.begin(), labels.end(), Label::ab_ab::a0b0_a2b1) != labels.end()
		|| std::find(labels.begin(), labels.end(), Label::ab_ab::a0b0_a2b2) != labels.end();
	const bool need_b_transpose = std::find(labels.begin(), labels.end(), Label::ab_ab::a0b0_a1b2) != labels.end();
	Ds_map Ds_a_transpose, Ds_b_transpose;
	if(need_a_transpose)
		Ds_a_transpose = LRI_Cal_Aux::cal_Ds_transpose(pack_a.Ds_ab);
	if(need_b_transpose)
	{
		if(need_a_transpose && name_a == name_b)
			Ds_b_transpose = Ds_a_transpose;
		else
			Ds_b_transpose = LRI_Cal_Aux::cal_Ds_transpose(pack_b.Ds_ab);
	}

	std::vector<std::map<TA,omp_lock_t>> locks;
	locks.reserve(Ds_results.size());
	for(Tensor_map &result : Ds_results)
		locks.push_back(LRI_Cal_Aux::init_lock_result(labels, this->parallel->list_A, result));

#ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
#endif

	#pragma omp parallel
	{
		std::vector<Ds_map> Ds_results_thread(Ds_results.size());
		for(const Label::ab_ab &label : labels)
		{
			const auto &list_A = this->parallel->list_A.at(Label_Tools::to_Aab_Aab(label));
			const std::vector<TA> list_Aa01 = LRI_Cal_Aux::filter_list_map(list_A.a01, pack_a.Ds_ab);
			const std::vector<TAC> list_Ab01 = LRI_Cal_Aux::filter_list_map(list_A.b01, pack_b.Ds_ab);
			const std::vector<TAC> list_Aa2 = LRI_Cal_Aux::filter_list_set(list_A.a2, pack_a.index_Ds_ab[0]);
			const std::vector<TAC> list_Ab2 = LRI_Cal_Aux::filter_list_set(list_A.b2, pack_b.index_Ds_ab[0]);

			switch(label)
			{
				case Label::ab_ab::a0b0_a1b1:
				{
					for(const TAC &Aa2 : list_Aa2)
					{
						if(this->filter_atom->filter_for1(label,Aa2))	continue;
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa2,Ab01))	continue;
							std::vector<Tensor<Tdata>> D_mul(Ds_results.size());
							for(const TA &Aa01 : list_Aa01)
							{
								if(this->filter_atom->filter_for31(label,Aa2,Ab01,Aa01))	continue;
								const Tensor<Tdata> &A = tools.front().get_Ds_ab(Label::ab::a,Aa01,Aa2);
								if(A.empty())	continue;
								const Tensor<Tdata> &M = tools.front().get_Ds_ab(Label::ab::a0b0,Aa01,Ab01);
								if(M.empty())	continue;
								std::vector<const Tensor<Tdata> *> S(Ds_results.size(), nullptr);
								bool any = false;
								for(std::size_t s=0; s<S.size(); ++s)
								{
									S[s] = &tools[s].get_Ds_ab(Label::ab::a1b1,Aa01,Ab01);
									any = any || !S[s]->empty();
								}
								if(!any)	continue;
								const Tensor<Tdata> L = Tensor_Multiply::x1x2y1_ax1x2_ay1(A,M);
								for(std::size_t s=0; s<S.size(); ++s)
									if(!S[s]->empty())
										LRI_Cal_Aux::add_Ds(Tensor_Multiply::x1x2y1_ax1x2_ay1(L,*S[s]),D_mul[s]);
							}
							bool any = false;
							for(const auto &D : D_mul) any = any || !D.empty();
							if(!any)	continue;
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for32(label,Aa2,Ab01,Ab2))	continue;
								const Tensor<Tdata> &B = tools.front().get_Ds_ab(Label::ab::b,Ab01,Ab2);
								if(B.empty())	continue;
								for(std::size_t s=0; s<D_mul.size(); ++s)
									if(!D_mul[s].empty())
									{
										Tensor<Tdata> out = Tensor_Multiply::x0y2_x0ab_aby2(D_mul[s],B);
										LRI_Cal_Aux::add_Ds(std::move(out),Ds_results_thread[s][Aa2.first][{Ab2.first,(Ab2.second-Aa2.second)%this->period}]);
									}
							}
						}
						for(std::size_t s=0; s<Ds_results.size(); ++s)
							LRI_Cal_Aux::add_Ds_omp_try_map(Ds_results_thread[s],Ds_results[s],locks[s],fac_add_Ds);
					}
				} break;

				case Label::ab_ab::a0b0_a1b2:
				{
					for(const TAC &Ab01 : list_Ab01)
					{
						if(this->filter_atom->filter_for1(label,Ab01))	continue;
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for2(label,Ab01,Aa01))	continue;
							const Tensor<Tdata> &M = tools.front().get_Ds_ab(Label::ab::a0b0,Aa01,Ab01);
							if(M.empty())	continue;
							std::vector<std::map<TAC,Tensor<Tdata>>> Ds_result_fixed(Ds_results.size());
							std::vector<Tensor<Tdata>> D_mul(Ds_results.size());
							for(std::size_t s=0; s<Ds_results.size(); ++s)
							{
								for(const TAC &Ab2 : list_Ab2)
								{
									if(this->filter_atom->filter_for31(label,Ab01,Aa01,Ab2))	continue;
									const Tensor<Tdata> &Bt = Global_Func::find(Ds_b_transpose,Ab01.first,TAC{Ab2.first,(Ab2.second-Ab01.second)%this->period});
									const Tensor<Tdata> &S = tools[s].get_Ds_ab(Label::ab::a1b2,Aa01,Ab2);
									if(Bt.empty() || S.empty())	continue;
									LRI_Cal_Aux::add_Ds(Tensor_Multiply::x0x1y0_x0x1a_y0a(Bt,S),D_mul[s]);
								}
							}
							bool any = false;
							for(const auto &D : D_mul) any = any || !D.empty();
							if(!any)	continue;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for32(label,Ab01,Aa01,Aa2))	continue;
								const Tensor<Tdata> &A = tools.front().get_Ds_ab(Label::ab::a,Aa01,Aa2);
								if(A.empty())	continue;
								const Tensor<Tdata> L = Tensor_Multiply::x1y1y2_ax1_ay1y2(M,A);
								for(std::size_t s=0; s<D_mul.size(); ++s)
								if(!D_mul[s].empty())
									{
										Tensor<Tdata> out = Tensor_Multiply::x2y0_abx2_y0ab(L,D_mul[s]);
										LRI_Cal_Aux::add_Ds(std::move(out),Ds_result_fixed[s][Aa2]);
									}
							}
							for(std::size_t s=0; s<Ds_results.size(); ++s)
							{
								if(!Ds_result_fixed[s].empty())
									LRI_Cal_Aux::add_Ds(LRI_Cal_Aux::Ds_exchange(std::move(Ds_result_fixed[s]),Ab01,this->period),Ds_results_thread[s]);
								LRI_Cal_Aux::add_Ds_omp_try_map(Ds_results_thread[s],Ds_results[s],locks[s],fac_add_Ds);
							}
						} // Aa01
					}
				} break;

				case Label::ab_ab::a0b0_a2b1:
				{
					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
							std::vector<Tensor<Tdata>> D_mul(Ds_results.size());
							std::vector<std::map<TAC,Tensor<Tdata>>> Ds_result_fixed(Ds_results.size());
							for(std::size_t s=0; s<Ds_results.size(); ++s)
								for(const TAC &Aa2 : list_Aa2)
								{
									if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
									const Tensor<Tdata> &At = Global_Func::find(Ds_a_transpose,Aa01,Aa2);
									const Tensor<Tdata> &S = tools[s].get_Ds_ab(Label::ab::a2b1,Aa2,Ab01);
									if(At.empty() || S.empty())	continue;
									LRI_Cal_Aux::add_Ds(Tensor_Multiply::x0x1y1_x0x1a_ay1(At,S),D_mul[s]);
								}
							bool any = false;
							for(const auto &D : D_mul) any = any || !D.empty();
							if(any)
							{
								const Tensor<Tdata> &M = tools.front().get_Ds_ab(Label::ab::a0b0,Aa01,Ab01);
								if(!M.empty())
									for(const TAC &Ab2 : list_Ab2)
									{
										if(this->filter_atom->filter_for32(label,Aa01,Ab01,Ab2))	continue;
											const Tensor<Tdata> &B = tools.front().get_Ds_ab(Label::ab::b,Ab01,Ab2);
										if(B.empty())	continue;
										const Tensor<Tdata> R = Tensor_Multiply::x0y1y2_x0a_ay1y2(M,B);
										for(std::size_t s=0; s<D_mul.size(); ++s)
											if(!D_mul[s].empty())
											{
												Tensor<Tdata> out = Tensor_Multiply::x0y2_x0ab_aby2(D_mul[s],R);
													LRI_Cal_Aux::add_Ds(std::move(out),Ds_result_fixed[s][Ab2]);
												}
										}
								}
								for(std::size_t s=0; s<Ds_results.size(); ++s)
									if(!Ds_result_fixed[s].empty())
										LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed[s]),Ds_results_thread[s][Aa01]);
								for(std::size_t s=0; s<Ds_results.size(); ++s)
								LRI_Cal_Aux::add_Ds_omp_try_map(Ds_results_thread[s],Ds_results[s],locks[s],fac_add_Ds);
						} // Ab01
					}
				} break;

				case Label::ab_ab::a0b0_a2b2:
				{
					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
						{
							const TAC &Ab2 = list_Ab2[ib2];
							if(this->filter_atom->filter_for2(label,Aa01,Ab2))	continue;
							std::vector<Tensor<Tdata>> D_mul(Ds_results.size());
							std::vector<std::map<TAC,Tensor<Tdata>>> Ds_result_fixed(Ds_results.size());
							for(std::size_t s=0; s<Ds_results.size(); ++s)
								for(const TAC &Aa2 : list_Aa2)
								{
									if(this->filter_atom->filter_for31(label,Aa01,Ab2,Aa2))	continue;
									const Tensor<Tdata> &At = Global_Func::find(Ds_a_transpose,Aa01,Aa2);
									const Tensor<Tdata> &S = tools[s].get_Ds_ab(Label::ab::a2b2,Aa2,Ab2);
									if(At.empty() || S.empty())	continue;
									LRI_Cal_Aux::add_Ds(Tensor_Multiply::x0x1y1_x0x1a_ay1(At,S),D_mul[s]);
								}
							bool any = false;
							for(const auto &D : D_mul) any = any || !D.empty();
							if(any)
							{
								for(const TAC &Ab01 : list_Ab01)
								{
									if(this->filter_atom->filter_for32(label,Aa01,Ab2,Ab01))	continue;
									const Tensor<Tdata> &M = tools.front().get_Ds_ab(Label::ab::a0b0,Aa01,Ab01);
									if(M.empty())	continue;
									const Tensor<Tdata> &B = tools.front().get_Ds_ab(Label::ab::b,Ab01,Ab2);
									if(B.empty())	continue;
									const Tensor<Tdata> Rraw = Tensor_Multiply::x0y1y2_x0a_ay1y2(M,B);
											const Tensor<Tdata> R = LRI_Cal_Aux::tensor3_transpose(Rraw);
									for(std::size_t s=0; s<D_mul.size(); ++s)
										if(!D_mul[s].empty())
										{
												Tensor<Tdata> out = Tensor_Multiply::x0y0_x0ab_y0ab(D_mul[s],R);
													LRI_Cal_Aux::add_Ds(std::move(out),Ds_result_fixed[s][Ab01]);
												}
								}
							}
							for(std::size_t s=0; s<Ds_results.size(); ++s)
									if(!Ds_result_fixed[s].empty())
										LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed[s]),Ds_results_thread[s][Aa01]);
								for(std::size_t s=0; s<Ds_results.size(); ++s)
								LRI_Cal_Aux::add_Ds_omp_try_map(Ds_results_thread[s],Ds_results[s],locks[s],fac_add_Ds);
						} // Ab2
					}
				} break;
				default:
					break;
			}
		}
		for(std::size_t s=0; s<Ds_results.size(); ++s)
			LRI_Cal_Aux::add_Ds_omp_wait_map(Ds_results_thread[s],Ds_results[s],locks[s],fac_add_Ds);
	}

	for(std::size_t s=0; s<Ds_results.size(); ++s)
		LRI_Cal_Aux::destroy_lock_result(locks[s],Ds_results[s]);

#ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
	#endif
	std::map<std::string,Tensor_map> results;
	for(std::size_t s=0; s<result_names.size(); ++s)
		results.emplace(result_names[s],std::move(Ds_results[s]));
	results_out = std::move(results);
}

}
