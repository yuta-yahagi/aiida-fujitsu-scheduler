#!/usr/bin/env python
"""Run a test calculation on localhost.

Usage: ./example_01.py
"""

from os import path

import click
from aiida import cmdline, engine, orm, load_profile

INPUT_DIR = path.join(path.dirname(path.realpath(__file__)), "input_files")


def test_run(add_code=None):
    """Run a calculation on the localhost computer.

    Uses test helpers to create AiiDA Code on the fly.
    """
    if add_code is None:
        # get code
        add_code = orm.load_code("add@wisteria-o")

    # Prepare input parameters
    builder = add_code.get_builder()
    builder.x = orm.Int(2)
    builder.y = orm.Int(3)
    builder.metadata.options.max_memory_kb = 1024
    builder.metadata.options.withmpi = True
    builder.metadata.options.resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1
    }
    builder.metadata.options.account = "gd55"
    builder.metadata.options.queue_name = "debug-o"

    # Note: in order to submit your calculation to the aiida daemon, do:
    # from aiida.engine import submit
    # future = submit(CalculationFactory('fujitsu_scheduler'), **inputs)
    result,pk = engine.run_get_pk(builder)
    print(f"pk = <{pk}>")
    print(f"results = \n{result}")


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODE()
def cli(code):
    """Run example.

    Example usage: $ ./example_01.py --code diff@localhost

    Alternative (creates diff@localhost-test code): $ ./example_01.py

    Help: $ ./example_01.py --help
    """
    test_run(code)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
