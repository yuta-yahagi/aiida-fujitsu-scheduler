#!/usr/bin/env python
"""Run a test calculation on localhost.

Usage: ./example_01.py
"""

from os import path

import click
from aiida import cmdline, engine, orm, load_profile
from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group
from ase.build import bulk

INPUT_DIR = path.join(path.dirname(path.realpath(__file__)), "input_files")

load_profile()

def test_run(add_code=None):
    """Run a calculation on the localhost computer.

    Uses test helpers to create AiiDA Code on the fly.
    """
    if add_code is None:
        # get code
        add_code = orm.load_code("pw@wisteria-o")

    # Prepare input parameters
    builder = add_code.get_builder()

    structure = StructureData(ase=bulk('Si', 'fcc', 5.43))
    builder.structure = structure

    pseudo_family = load_group('SSSP/1.1/PBE/efficiency')
    builder.pseudos = pseudo_family.get_pseudos(structure=structure)

    cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(
    structure=structure,
    unit='Ry'
)

    # 必要な制御パラメータをビルダーにセットする。
    parameters = Dict({
        'CONTROL': {
            'calculation': 'scf'
        },
        'SYSTEM': {
            'ecutwfc': cutoff_wfc,
            'ecutrho': cutoff_rho,
        }
    })
    builder.parameters = parameters

    # K点として、2x2x2 Monkhorst-Pack グリッドを生成し、ビルダーにセットする。
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2])
    builder.kpoints = kpoints
        
    # builder.metadata.options.max_memory_kb = 1024*1024
    builder.metadata.options.withmpi = False # MPIを使用しない
    builder.metadata.options.resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
    }
    builder.metadata.options.account = "gd55"
    builder.metadata.options.queue_name = "debug-o"

    # Note: in order to submit your calculation to the aiida daemon, do:
    # from aiida.engine import submit
    # future = submit(CalculationFactory('fujitsu_scheduler'), **inputs)
    
    engine.run(builder)
    # result = engine.submit(builder)
    # print(f"Submitted calculation with PK={result.pk}, UUID={result.uuid}")


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
