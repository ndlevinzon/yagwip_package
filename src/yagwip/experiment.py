from yagwip.gromacs_sim import GromacsSim, correct_folder
import os
from shutil import copyfile, copytree, rmtree
import glob
import re


class Experiment:
    """ An Experiment is a group of GROMACS simulations run with identical conditions.
    """

    def __init__(self, pdb_file_list, maindir, basefilesdir, jobdir, n_replicas, ffcode, solcode, water,
                 job_header_script=None, job_header_run=None, mode='default'):
        if mode != 'default':
            raise ValueError('No other modes than default currently implemented...')
        if job_header_script is None:
            job_header_script = get_default_job_header('3:00:00')
        if job_header_run is None:
            job_header_run = get_default_job_header('3:00:00')
        self.header_script = job_header_script
        self.header_run = job_header_run
        self.files = pdb_file_list
        self.maindir = correct_folder(maindir)
        self.basefiles = glob.glob('{0}/*.mdp'.format(correct_folder(basefilesdir)))
        self.forcefields = glob.glob('{0}/*.ff'.format(correct_folder(basefilesdir)))
        self.jobdir = correct_folder(jobdir)
        self.nreps = n_replicas
        self.ffcode = ffcode
        self.solcode = solcode
        self.water = water
        self.dirs = []
        self.list_dirs_copy_pdbs()

    def list_dirs_copy_pdbs(self):
        for filename in self.files:
            assert filename.endswith('.pdb')
            name = filename.split('/')[-1][:-4]
            for repl in range(1, self.nreps + 1):
                newdir = self.maindir + '/' + name + '_{}'.format(repl)
                self.dirs.append(newdir)
                copyfile(filename, newdir + '/' + name + '_{}.pdb'.format(repl))

    def initialize_dirs_copy_basefiles(self):
        for dirname in self.dirs:
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            for to_delete in os.listdir(dirname):
                to_delete_path = os.path.join(dirname, to_delete)
                if not os.path.isdir(to_delete_path):
                    os.remove(to_delete_path)
                else:
                    rmtree(to_delete_path)

            for basefile in self.basefiles:
                copyfile(basefile, dirname + '/{0}'.format(basefile.split('/')[-1]))

            for ffpath in self.forcefields:
                ffdest = '{0}/{1}'.format(dirname, ffpath.split('/')[-1])
                if not os.path.isdir(ffdest):
                    copytree(ffpath, ffdest)

    def create_all_scripts(self):
        self.create_setup_scripts()
        # self.create_mdrun_jobs("run_primer")
        self.create_mdrun_jobs("run", extend=True)

    def create_setup_scripts(self):
        with open(self.jobdir + '/start_setup.sh', 'w') as f:
            for dirname in self.dirs:
                create_setup_script(dirname, self.ffcode, self.solcode, self.water)
                f.write('sbatch {0}\n'.format(create_script_job(dirname, self.jobdir, self.header_script)))

    def create_mdrun_jobs(self, prefix, extend=False, chain_len=8):
        with open(self.jobdir + '/start_{0}_jobs.sh'.format(prefix), 'w') as starter_f:
            for dirname in self.dirs:
                prec_name = None
                for i in range(chain_len):
                    cur_suffix = ""
                    if i == 0:
                        cur_suffix = "primer"
                    else:
                        cur_suffix = "{}".format(i)
                    cur_name = create_mdrun_job(dirname, self.jobdir, self.header_run, prefix=prefix,
                                                suffix=cur_suffix, extend=extend)
                    if i == 0:
                        starter_f.write('sbatch {0}\n'.format(cur_name))
                    else:
                        with open(self.jobdir + "/{0}".format(prec_name), "a") as chaining_f:
                            chaining_f.write('sbatch {0}/{1}\n'.format(self.jobdir, cur_name))
                    prec_name = cur_name


def create_setup_script(dirname, ffcode, solcode, water):
    with open(dirname + '/' + 'setup_sim.py', 'w') as f:
        f.write("""from yagpyw.gromacs_sim import GromacsSim

if __name__ == "__main__":
    sim = GromacsSim('.', '{0}', 'gmx')
    sim.clean_all_except()
    sim.pdb2gmx('{1}', water='{2}')
    sim.solvate(water_model='{2}')
    sim.genion('{3}', grompp_options='-maxwarn 1')
    sim.em(mdpfile='min.mdp')
    sim.nvt()
    sim.npt()
    sim.prepare_run(mdpfile='md.mdp', outname='md')
""".format(dirname.split('/')[-1], re.sub('\n', '\\\\n', ffcode), water, re.sub('\n', '\\\\n', solcode)))


def create_script_job(dirname, jobsdir, header):
    name = dirname.split('/')[-1]
    jobname = "setup_job_{0}.sh".format(name)
    with open(jobsdir + "/" + jobname, "w") as f:
        f.write(header)
        f.write('\n')
        f.write('cd {0}\n'.format(dirname))
        f.write('python setup_sim.py\n')
    return jobname


def create_mdrun_job(dirname, jobsdir, header, prefix, suffix, extend=False):
    name = dirname.split('/')[-1]
    extend_str = ""
    if extend:
        extend_str = "-cpi {0}.md_prev ".format(name)
    jobname = "{0}_{1}_{2}.sh".format(prefix, name, suffix)
    with open(jobsdir + "/" + jobname, "w") as f:
        f.write(header)
        f.write('\n')
        f.write('cd {0}\n'.format(dirname))
        # past command, with MPI. Made 8-CPUs processes crash most of the time (but not always, weird)
        # f.write('mpirun gmx_mpi mdrun -deffnm {0}.md {1}-maxh 3 -nsteps -1\n'.format(name, extend_str))

        # new command, no MPI, almost same performance and never crashes (so far)
        f.write('gmx mdrun -deffnm {0}.md {1}-maxh 3 -nsteps -1\n'.format(name, extend_str))
    return jobname


def get_default_job_header(time_str, tasks_per_node=8, cpus_per_task=5, ncpus=40,
                           py_env='/home/mailhoto/py_env'):
    assert tasks_per_node*cpus_per_task == ncpus
    return """#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2048M
#SBATCH --time={0}
#SBATCH --account=rrg-najmanov
module purge
module load gcc/7.3.0 openmpi/3.1.2 gromacs/2019.3
source {1}/bin/activate
""".format(time_str, py_env)

# salloc --ntasks=8 --cpus-per-task=1 --nodes=1 --mem-per-cpu=2048M --account=rrg-najmanov --time=3:00:00




