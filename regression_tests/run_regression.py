#!/usr/bin/env python3
"""Driver for running regression test for LibRPA
"""
import contextlib
import pathlib
import sys
from backend.commandparser import get_parser
from backend.xmlparser import XMLParser
from backend.driver import TestDriver


class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _run(args, parser):

    if args.mode is None:
        parser.print_help()
        return 1

    # Parse the test suite
    suite = XMLParser(args.xml)

    # Create the test driver
    driver = TestDriver(args.dir_input, args.dir_ref,
                        args.workspace, suite.groups)

    # Initialize workspace and run the tests
    try:
        driver.initialize(args.ntasks, args.nthreads, args.use_libri,
                          args.only, args.exclude)
    except ValueError as exc:
        parser.error(str(exc))

    status = 0

    if args.mode == "list":
        driver.list()
        return status

    if args.mode in ["run", "full"]:
        try:
            driver.run(args.librpa_exec, args.mpiexec, args.force, args.verbose)
        except ValueError as exc:
            parser.error(str(exc))

    if args.mode in ["analyze", "full"]:
        status = driver.analyze()
        driver.print()
        print()
        if status == 0:
            print("All selected tests PASSED :)")
        else:
            print("Some tests FAILED, please check above details")

    return status


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.mode in ["analyze", "full"]:
        output_path = pathlib.Path(args.output)
        if output_path.parent != pathlib.Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as output:
            with contextlib.redirect_stdout(Tee(sys.stdout, output)):
                status = _run(args, parser)
    else:
        status = _run(args, parser)

    sys.exit(status)
