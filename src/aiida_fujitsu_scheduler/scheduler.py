# -*- coding: utf-8 -*-
"""Template for a scheduler plugin."""
import datetime
import logging
from typing import TYPE_CHECKING
import re

from aiida.common.escaping import escape_for_bash
from aiida.common import exceptions
from aiida.common.lang import type_check
from aiida.schedulers import Scheduler, SchedulerError, SchedulerParsingError
from aiida.schedulers.datastructures import JobInfo, JobResource, JobState
from aiida.schedulers.plugins.pbsbaseclasses import PbsJobResource

_LOGGER = logging.getLogger(__name__)


_MAP_SCHEDULER_AIIDA_STATUS = {
    'ACCEPT'        : JobState.QUEUED,
    'ACC'           : JobState.QUEUED,
    'QUEUED'        : JobState.QUEUED,
    'QUE'           : JobState.QUEUED,
    'RUNNING-A'     : JobState.RUNNING,
    'RNA'           : JobState.RUNNING,
    'RUNNING-P'     : JobState.RUNNING,
    'RNP'           : JobState.RUNNING,
    'RUNNING'       : JobState.RUNNING,
    'RUN'           : JobState.RUNNING,
    'RUNNING-E'     : JobState.RUNNING,
    'RNE'           : JobState.RUNNING,
    'RUNOUT'        : JobState.RUNNING,
    'RNO'           : JobState.RUNNING,
    'EXIT'          : JobState.DONE, 
    'EXT'           : JobState.DONE,
    'END'           : JobState.DONE, 
    'REJECT'        : JobState.DONE,
    'RJT'           : JobState.DONE,
    'CANCEL'        : JobState.DONE,
    'CCL'           : JobState.DONE,
    'HOLD'          : JobState.QUEUED_HELD,
    'HLD'           : JobState.QUEUED_HELD,
    'ERROR'         : JobState.QUEUED_HELD,
    'ERR'           : JobState.QUEUED_HELD,
}

class GPUJobResource(PbsJobResource):
    """Job resources class for GPU machine."""

    
    _default_fields = tuple(list(PbsJobResource._default_fields)+['num_gpu'])
    if TYPE_CHECKING:
        num_gpu: int

    @classmethod
    def validate_resources(cls, **kwargs):
        resources = super().validate_resources(**kwargs)
        num_gpu=kwargs.pop('num_gpu', None)
        if num_gpu is not None:
            if num_gpu > 0:
                setattr(resources, 'num_gpu', num_gpu)
        return resources


class FujitsuScheduler(Scheduler):
    """Base class template for a scheduler."""

    # Query only by list of jobs and not by user
    _features = {
        'can_query_by_user': False,
    }

    # The class to be used for the job resource.
    _job_resource_class = GPUJobResource # This needs to be set to a subclass of :class:`~aiida.schedulers.datastructures.JobResource`

    _map_status = _MAP_SCHEDULER_AIIDA_STATUS

    def _get_joblist_command(self, jobs=None, user=None):
        """The command to report full information on existing jobs.

        :return: a string of the command to be executed to determine the active jobs.
        """

        return 'pjstat -v'

    def _get_detailed_job_info_command(self, job_id):
        """Return the command to run to get the detailed information on a job,
        even after the job has finished.

        The output text is just retrieved, and returned for logging purposes.
        """
        # Max history days is 31
        return f'pjstat -H --hday 31 -s {escape_for_bash(job_id)}'# for instance f'tracejob -v {escape_for_bash(job_id)}'
        # return f'pjstat -H --hday 31 -s {job_id}'
    
    def _get_submit_script_header(self, job_tmpl):
        """Return the submit script final part, using the parameters from the job template.

        :param job_tmpl: a ``JobTemplate`` instance with relevant parameters set.
        """
        import string

        lines = []
        if job_tmpl.account:
            lines.append(f'#PJM -g {job_tmpl.account}')
        else:
            raise ValueError('Project code (job_tmpl.account) is required.')
        
        if job_tmpl.rerunnable:
            lines.append('#PJM --restart')
        else:
            lines.append('#PJM --norestart')

        if job_tmpl.email:
            lines.append(f'#PJM --mail-list {job_tmpl.email}')
        
        if job_tmpl.email_on_started:
            lines.append('#PJM -m=b')
        if job_tmpl.email_on_terminated:
            lines.append('#PJM -m=e')
        
        if job_tmpl.job_name:
            # I leave only letters, numbers, dots, dashes and underscores
            # Note: I don't compile the regexp, I am going to use it only once
            job_title = re.sub(r'[^a-zA-Z0-9_.-]+', '', job_tmpl.job_name)
            # Truncate to the first 128 characters
            # Nothing is done if the string is shorter.
            job_title = job_title[:128]
            lines.append(f'#PJM -N {job_title}')
        
        if job_tmpl.import_sys_environment:
            lines.append('#PJM -X')

        if job_tmpl.sched_output_path:
            lines.append(f'#PJM -o {job_tmpl.sched_output_path}')
        
        if job_tmpl.sched_error_path:
            lines.append(f'#PJM -e {job_tmpl.sched_error_path}')

        if job_tmpl.sched_join_files:
            lines.append('#PJM -j')
        
        if job_tmpl.queue_name:
            # Resource group (rscgrp) is identical to the queue name in Fujitsu system
            lines.append(f'#PJM -L rscgrp={job_tmpl.queue_name}')

        if job_tmpl.qos:
            self._logger.warning('qos is given as <{job_tmpl.qos}>, but, is not supported by the Fujitsu scheduler')

        if not job_tmpl.job_resource:
            raise ValueError('Job resources (as the num_machines) are required.')
        
        lines.append(f'#PJM -L node={job_tmpl.job_resource.num_machines}')
        if getattr(job_tmpl.job_resource, 'num_gpu', None):
            lines.append(f'#PJM -L gpu={job_tmpl.job_resource.num_gpu}')

        if job_tmpl.job_resource.tot_num_mpiprocs > 1:
            lines.append(f'#PJM --mpi proc={job_tmpl.job_resource.tot_num_mpiprocs}')
        
        if job_tmpl.job_resource.num_cores_per_machine is not None:
            num_cores_per_proc = job_tmpl.job_resource.num_cores_per_machine // job_tmpl.job_resource.num_mpiprocs_per_machine
            if num_cores_per_proc > 1:
                lines.append(f'#PJM --omp thread={num_cores_per_proc}')

        if job_tmpl.max_wallclock_seconds:
            lines.append(f'#PJM -L elapse={job_tmpl.max_wallclock_seconds}')

        # It is the memory per node, not per cpu!
        if job_tmpl.max_memory_kb:
            try:
                physical_memory_kb = int(job_tmpl.max_memory_kb)
                if physical_memory_kb <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(
                    f'max_memory_kb must be a positive integer (in kB)! It is instead `{job_tmpl.max_memory_kb}`'
                )
            # node-mem: Specify the real memory required per node in KiB.
            mem_MiB=physical_memory_kb//1024
            if mem_MiB < 1024:
                self.logger.info(f'Memory allocation should be learger than 1 GiB ({mem_MiB}MiB) -> set to 1 GiB')
                mem_MiB=1024

            lines.append(f'#PJM -L node-mem={mem_MiB}')

        if job_tmpl.custom_scheduler_commands:
            lines.append(job_tmpl.custom_scheduler_commands)
        
        return '\n'.join(lines)

    def _get_submit_command(self, submit_script):
        """Return the string to execute to submit a given script.

        .. warning:: the `submit_script` should already have been bash-escaped

        :param submit_script: the path of the submit script relative to the working directory.
        :return: the string to execute to submit a given script.
        """
        # -z jid option is used to get the job id only
        submit_command = f'pjsub -z jid {submit_script}' # for instance f'qsub {submit_script}'

        self.logger.info(f'submitting with: {submit_command}')

        return submit_command

    def _parse_joblist_output(self, retval, stdout, stderr):
        """Parse the joblist output as returned by executing the command returned by `_get_joblist_command` method.

        :return: list of `JobInfo` objects, one of each job each with at least its default params implemented.
        """
        # See discussion in _get_joblist_command on how we ensure that AiiDA can expect exit code 0 here.
        if retval != 0:
            raise SchedulerError(
                f"""squeue returned exit code {retval} (_parse_joblist_output function)
                    stdout='{stdout.strip()}'
                    stderr='{stderr.strip()}'
                """
            )
        if stderr.strip():
            self.logger.warning(
                f"squeue returned exit code 0 (_parse_joblist_output function) but non-empty stderr='{stderr.strip()}'"
            )

        lines=stdout.strip().splitlines()
        iheader=None
        for i, l in enumerate(lines):
            if 'JOB_ID' in l:
                iheader=i
                break
        if iheader is None:
            # No unfinished job
            return []
        headers=lines[iheader].split()
        jobdata_raw = []
        for l in lines[iheader+1:]:
            vals=re.split(r'\s{2,}', l)
            jobdata_raw.append(dict(zip(headers, vals)))

        job_list=[]
        for row in jobdata_raw:
            if len(row) != len(headers):
                raise SchedulerParsingError(f'Error parsing the output of the scheduler (length mismatch): {row}')
            
            this_job = JobInfo()
            try:
                this_job.job_id = row['JOB_ID']
            except KeyError:
                raise SchedulerParsingError(f'Error parsing the output of the scheduler (job_id): {row}')

            if row.get('STATUS') is not None:
                this_job.job_state = row['STATUS']
            elif row.get('ST') is not None:
                this_job.job_state = row['ST']
            else:
                this_job.job_state = JobState.UNDETERMINED
                self.logger.warning(f'Failed to parse STATUS/ST -> set to UNDETERMINED')

            map_scheduler_jobinfo_str = {
                'JOB_NAME'  : 'title',
                'RSCGROUP'  : 'queue_name',
                'GROUP'     : 'account',
                'USER'      : 'job_owner',
            }
            for key, value in map_scheduler_jobinfo_str.items():
                if row.get(key) is not None:
                    setattr(this_job, value, row[key])
                else:
                    self.logger.info(f'Failed to parse {key} -> set blank')
                    setattr(this_job, value, '')
            
            if row.get('START_DATE') is not None:
                time_obj = datetime.datetime.strptime(row['START_DATE'], '%m/%d %H:%M:%S')
                this_year=datetime.datetime.now().year
                this_job.dispatch_time = time_obj.replace(year=this_year)
            else:
                self.logger.info('Failed to parse START_DATE -> set to None')
                this_job.dispatch_time = None

            if row.get('ELAPSE') is not None:
                time_obj = datetime.datetime.strptime(row['ELAPSE'], '%H:%M:%S')
            elif row.get('ELAPSE_TIM') is not None:
                time_obj = datetime.datetime.strptime(row['ELAPSE_TIM'], '%H:%M:%S')
            else:
                time_obj = None
            
            if time_obj is not None:
                seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
                this_job.wallclock_time_seconds = seconds
            else:
                self.logger.info('Failed to parse ELAPSE/ELAPSE_TIM -> set to None')
                this_job.wallclock_time_seconds = None

            if row.get('NODE') is not None:
                this_job.num_machines = int(row['NODE'])
            else:
                self.logger.info('Failed to parse NODE -> set to None')
                this_job.num_machines = None

            job_list.append(this_job)
        return job_list

    def _parse_submit_output(self, retval, stdout, stderr):
        """Parse the output of the submit command returned by calling the `_get_submit_command` command.

        :return: a string with the job ID.
        """
        if retval != 0:
            self.logger.error(f'Error in _parse_submit_output: retval={retval}; stdout={stdout}; stderr={stderr}')
            raise SchedulerError(f'Error during submission, retval={retval}; stdout={stdout}; stderr={stderr}')

        if stderr.strip():
            self.logger.warning(f'in _parse_submit_output there was some text in stderr: {stderr}')

        if 'ERR' in stdout:
            self.logger.error(f'in _parse_submit_output: {stdout}')
            raise SchedulerError(f'Error during submission: {stdout}')

        if stdout.strip():
            # `pjsub -z jid` shuld returns the job id only if it works correctly.
            job_id = stdout.strip()
            self.logger.info(f'Job submitted with jobid {job_id}')
            return job_id
        else:
            self.logger.error(f'Error in _parse_submit_output: retval={retval}; stdout={stdout}; stderr={stderr}')
            raise SchedulerError(f'Error during submission, retval={retval}; stdout={stdout}; stderr={stderr}')

    def _get_kill_command(self, jobid):
        """Return the command to kill the job with specified jobid."""

        self.logger.info(f'killing job {jobid}')

        return f'pjdel {jobid}'  # for instance f'qdel {jobid}'

    def _parse_kill_output(self, retval, stdout, stderr):
        """Parse the output of the kill command.

        :return: True if everything seems ok, False otherwise.
        """
        if retval != 0:
            self.logger.error(f'Error in _parse_kill_output: retval={retval}; stdout={stdout}; stderr={stderr}')
            return False
        
        if 'ERR' in stdout:
            self.logger.warning(f'Something error in _parse_kill_output: retval={retval}; stdout={stdout}; stderr={stderr}')

        return True

    def parse_output(self, detailed_job_info, stdout, stderr):
        """Parse the output of the scheduler.

        :param detailed_job_info: dictionary with the output returned by the `Scheduler.get_detailed_job_info` command.
            This should contain the keys `retval`, `stdout` and `stderr` corresponding to the return value, stdout and
            stderr returned by the accounting command executed for a specific job id.
        :param stdout: string with the output written by the scheduler to stdout
        :param stderr: string with the output written by the scheduler to stderr
        :return: None or an instance of `aiida.engine.processes.exit_code.ExitCode`
        :raises TypeError or ValueError: if the passed arguments have incorrect type or value
        """
        from aiida.engine import CalcJob

        if detailed_job_info is not None:
            type_check(detailed_job_info, dict)

            try:
                detailed_stdout = detailed_job_info['stdout']
            except KeyError:
                raise ValueError('the `detailed_job_info` does not contain the required key `stdout`.')

            type_check(detailed_stdout, str)

            # The format of the detailed job info should be a multiline string, where the first line is the header, with
            # the labels of the projected attributes. The following line should be the values of those attributes for
            # the entire job. Any additional lines correspond to those values for any additional tasks that were run.
            lines = detailed_stdout.strip().splitlines()

            ibegin=None
            for i, l in enumerate(lines):
                if 'Information' in l:
                    ibegin=i
                    break
            if ibegin is None:
                raise ValueError('the `detailed_job_info.stdout` contained less than two lines.')

            lines=lines[ibegin+1:]
            exitline=None
            for l in lines:
                if l.startswith(' EXIT CODE '):
                    break
            if exitline is None:
                raise ValueError('the `detailed_job_info.stdout` does not contain the exit code.')

            exit_code=int(exitline.split(':')[1].strip())

            if exit_code == 29:
                return CalcJob.exit_codes.ERROR_SCHEDULER_NODE_FAILURE
            if exit_code == 11:
                return CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME
            if exit_code == 12:
                return CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY
            return None
