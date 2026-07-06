#include "input_parser.h"

#include <regex>

static const std::string SPACE_SEP = "[ \r\f\t]*";

const std::string InputParser::KV_SEP = "=";
const std::string InputParser::COMMENTS_IDEN = "[#!]";

static std::string get_last_matched(const std::string &s, const std::string &key,
                                    const std::string &vregex, int igroup)
{
    std::string sout = "";
    std::regex r("(^|[\r\n])" + SPACE_SEP + key + SPACE_SEP
                 + InputParser::KV_SEP + SPACE_SEP + vregex,
                 std::regex_constants::ECMAScript | std::regex_constants::icase);
    std::sregex_iterator si(s.begin(), s.end(), r);
    auto ei = std::sregex_iterator();
    for (auto i = si; i != ei; i++)
    {
        std::smatch match = *i;
        // std::cout << "whole matched string: " << match.str(0) << std::endl;
        sout = match.str(igroup + 1);
    }
    return sout;
}

void InputParser::parse_double(const std::string &vname, double &var, int &flag) const
{
    flag = 0;
    std::string s = get_last_matched(params, vname, "(-?[\\d]+\\.?([\\d]+)?([ed]-?[\\d]+)?)", 1);
    if (s != "")
    {
        try
        {
            var = std::stod(s);
        }
        catch (std::invalid_argument const&)
        {
            flag = 2;
        }
    }
    else
        flag = 1;
}


void InputParser::parse_double(const std::string &vname, double &var, double de, int &flag) const
{
    parse_double(vname, var, flag);
    if (flag) var = de;
}

void InputParser::parse_int(const std::string &vname, int &var, int &flag) const
{
    flag = 0;
    std::string s = get_last_matched(params, vname, "(-?[\\d]+)", 1);
    if (s != "")
    {
        try
        {
            var = std::stoi(s);
        }
        catch (std::invalid_argument const&)
        {
            flag = 2;
        }
    }
    else
        flag = 1;
}

void InputParser::parse_int(const std::string &vname, int &var, int de, int &flag) const
{
    parse_int(vname, var, flag);
    if (flag) var = de;
}

void InputParser::parse_string(const std::string &vname, std::string &var, int &flag) const
{
    flag = 0;
    std::string s = get_last_matched(params, vname, "([\\w\\d_\\- ,:;./]+)", 1);
    if (s != "")
        var = s;
    else
        flag = 1;
}

void InputParser::parse_string(const std::string &vname, std::string &var, const std::string &de, int &flag) const
{
    parse_string(vname, var, flag);
    if (flag) var = de;
}

void InputParser::parse_bool(const std::string &vname, bool &var, int &flag) const
{
    flag = 0;
    std::string s = get_last_matched(params, vname, "([\\w]+)", 1);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "true" || s == "t" || s == ".t.")
        var = true;
    else if (s == "f" || s == "false" || s == ".f.")
        var = false;
    else
        flag = 1;
}

void InputParser::parse_bool(const std::string &vname, bool &var, const bool &de, int &flag) const
{
    parse_bool(vname, var, flag);
    if (flag) var = de;
}

