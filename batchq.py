import subprocess
import os
import tempfile
import shlex
import re

sbatch_template = """#!/bin/bash

#SBATCH --job-name={jobname}
#SBATCH --output={log}
#SBATCH --error={log}
#SBATCH --account=pi-lgrandi
#SBATCH --qos={qos}
#SBATCH --partition={partition}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={cpus_per_task}

source activate {env}

#export PYTHONPATH="/home/ershockley/software:$PYTHONPATH"

{job}
"""

def submit_job(jobstring, log=None, partition='xenon1t', qos='xenon1t', account='pi-lgrandi',
               jobname='somejob', env='pax_head', delete_file=True, dry_run=False, mem_per_cpu=1000,
               cpus_per_task=1):

    if not log:
        log = os.path.join(os.environ['HOME'], 'job.log')

    sbatch_script = sbatch_template.format(jobname=jobname, log=log, qos=qos, partition=partition, account=account,
                                           env=env, job=jobstring, mem_per_cpu=mem_per_cpu,
                                           cpus_per_task=cpus_per_task)
    if env == "None":
        re.sub("source activate None", "", sbatch_script)

    _, file = tempfile.mkstemp(suffix='.sbatch')
    
    with open(file, 'w') as f:
        f.write(sbatch_script)

    if dry_run:
        print("sbatch script at %s:" % file)
        print(sbatch_script)
        return

    command = "sbatch %s" % file
    if not delete_file:
        print("Executing: %s" % command )
    subprocess.Popen(shlex.split(command)).communicate()
    #return subprocess.check_output(command, stderr=subprocess.STDOUT)

    if delete_file:
        os.remove(file)



def test():
    job = "sleep 10s; echo 'This is a test'"
    submit_job(job)

if __name__ == '__main__': test()
