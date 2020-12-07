import subprocess
import shlex
import glob
import os
import re


class GromacsSim:
    """ Represents one gromacs simulation, starting with a directory containing the initial PDB file. All the necessary
        steps (solvation, box definition, equilibration etc. can be automated. The debug mode allows the user to go
        through all the steps once, and then all the interactive commands are automated according to what was entered.
    """

    def __init__(self, basedir, basename, gmx_path, debug_mode=False):
        self.basedir = correct_folder(basedir)
        self.basename = basename
        self.gmx_path = gmx_path
        self.debug_mode = debug_mode

    def debug_on(self):
        self.debug_mode = True

    def debug_off(self):
        self.debug_mode = False

    def print_alloc(self):
        print("salloc --nodes=1 --ntasks-per-node=8 --cpus-per-task=5 --exclusive --mem=0 --time=3:00:00 --account=rrg-najmanov")

    def print_modules(self):
        print("""module purge
module load gcc/7.3.0 openmpi/3.1.2 gromacs/2019.3
module load python/3""")

    def execute(self, cmds_list, pipe_codes=None, workdir=None):
        """ Executes a list of commands in the simulation directory.
            - cmds_list is a list of full string commands.
            - pipecodes are the STDIN inputs to give to each command, as a corresponding list to cmds_list (if no inputs
            are needed, pipecodes can be left as None.
            - workdir will be the simulation directory if left to None
        """
        pipename = "pipefile.pipe"
        shname = "shcommand.sh"
        if workdir is None:
            workdir = self.basedir
        workdir = correct_folder(workdir)
        if not isinstance(cmds_list[0], str):
            raise TypeError("GromacsSim.execute() being called on object other than list of strings (commands), " +
                            "directory {0}".format(self.basedir))
        if pipe_codes is None:
            pipe_codes = [None] * len(cmds_list)
        if len(cmds_list) != len(pipe_codes):
            raise ValueError("GromacsSim.execute() called with len(pipecodes) != len(cmds_list), directory {0}".format(
                self.basedir))
        if self.debug_mode:
            return self.execute_debug(cmds_list, pipe_codes, workdir)
        result = None
        for (cmd, pc) in zip(cmds_list, pipe_codes):
            cmd_l = shlex.split(cmd) # list of strings instead of single string command
            if pc is None:
                result = subprocess.run(cmd_l, cwd=workdir, capture_output=True)
            else:
                write_pipefile(workdir, pc, pipename)
                cmd_l = cmd_l + ["<", pipename]
                write_sh(workdir, cmd_l, shname)
                cmd_l = ["sh", shname]
                result = subprocess.run(cmd_l, cwd=workdir, capture_output=True)
        return result

    def execute_debug(self, cmds_list, pipecodes, workdir):
        for cmd in cmds_list:
            print(cmd)

    def clean_all_except(self, files_list=None):
        """ Used to clean the simulation directory. By default, will remove everything except the initial .pdb,
            .mdp, .py and .ff files.
        """
        if files_list is None:
            files_list = ["{0}/{1}.pdb".format(self.basedir, self.basename)]
            files_list += glob.glob("{0}/*.mdp".format(self.basedir))
            files_list += glob.glob("{0}/*.py".format(self.basedir))
            files_list += glob.glob("{0}/*.ff".format(self.basedir))
        files_set = set(files_list)
        cur_files = glob.glob("{0}/*".format(self.basedir))
        for f in cur_files:
            if f not in files_set:
                s = "rm -r {0}".format(f)
                self.execute([s])

    def pdb2gmx(self, pc, water="spce", opt_args=" -ignh", pdb_fn=None):
        if pdb_fn is None:
            self.format = "{0}.pdb".format(self.basename)
            pdb_fn = self.format
        opt_args = correct_opt_arg(opt_args)
        gro_fn = "{0}.gro".format(self.basename)
        s = "{0} pdb2gmx -f {1} -o {2} -water {3}{4}".format(self.gmx_path, pdb_fn, gro_fn, water, opt_args)
        return self.execute([s], pipe_codes=[pc])

    def solvate(self, box_options=" -c -d 1.0 -bt cubic", water_model="spc216.gro"):
        box_options = correct_opt_arg(box_options)
        n1 = "{0}.gro".format(self.basename)
        n2 = "{0}.newbox.gro".format(self.basename)
        n3 = "{0}.solv.gro".format(self.basename)
        s = "{0} editconf -f {1} -o {2}{3}".format(self.gmx_path, n1, n2, box_options)
        s2 = "{0} solvate -cp {1} -cs {2} -o {3} -p topol.top".format(self.gmx_path, n2, water_model, n3)
        return self.execute([s, s2])

    def genion(self, sol_code, ion_options=" -pname NA -nname CL -conc 0.100 -neutral", grompp_options=""):
        ion_options = correct_opt_arg(ion_options)
        grompp_options = correct_opt_arg(grompp_options)
        n1 = "{0}.solv.gro".format(self.basename)
        n2 = "{0}.solv.ions.gro".format(self.basename)
        s = "{0} grompp -f ions.mdp -c {1} -r {1} -p topol.top -o ions.tpr {2}"\
            .format(self.gmx_path, n1, grompp_options)
        s2 = "{0} genion -s ions.tpr -o {1} -p topol.top{2}".format(self.gmx_path, n2, ion_options)
        pipe_codes = [None, sol_code]
        return self.execute([s, s2], pipe_codes=pipe_codes)

    def em(self, mdpfile="minim.mdp", suffix=".solv.ions", tprname="em", prefix = "mpirun ", mdrun_suffix=""):
        mdrun_suffix = correct_opt_arg(mdrun_suffix)
        s = "{0} grompp -f {1} -c {2}{3}.gro -r {2}{3}.gro -p topol.top -o {2}.{4}.tpr"\
            .format(self.gmx_path, mdpfile, self.basename, suffix, tprname)
        s2 = prefix + "{0}_mpi mdrun -deffnm {1}.{2}{3}".format(self.gmx_path, self.basename, tprname, mdrun_suffix)
        return self.execute([s, s2])

    def nvt(self, mdpfile="nvt.mdp", suffix=".em", tprname="nvt", mdrun_suffix=""):
        mdrun_suffix = correct_opt_arg(mdrun_suffix)
        return self.em(mdpfile, suffix, tprname, mdrun_suffix=mdrun_suffix)

    def npt(self, mdpfile="npt.mdp", suffix=".nvt", tprname="npt", mdrun_suffix=""):
        mdrun_suffix = correct_opt_arg(mdrun_suffix)
        return self.em(mdpfile, suffix, tprname, mdrun_suffix=mdrun_suffix)

    def production(self, mdpfile="md1ns.mdp", inputname="npt.", outname="md1ns", prefix="mpirun ", mdrun_suffix=""):
        mdrun_suffix = correct_opt_arg(mdrun_suffix)
        s = "{0} grompp -f {1} -c {2}.{3}gro -r {2}.{3}gro -p topol.top -o {2}.{4}.tpr"\
            .format(self.gmx_path, mdpfile, self.basename, inputname, outname)
        s2 = prefix + "{0}_mpi mdrun -deffnm {1}.{2}{3}".format(self.gmx_path, self.basename, outname, mdrun_suffix)
        return self.execute([s, s2])

    def production_finished(self, mdname="md1ns"):
        logfile = "{0}/{1}.{2}.log".format(self.basedir, self.basename, mdname)
        if os.path.isfile(logfile):
            with open(logfile) as f:
                lines = f.readlines()
            if lines[-2].startswith("Finished mdrun"):
                return True
        return False

    def prepare_run(self, mdpfile="md1ns.mdp", inputname="npt.", outname="md1ns"):
        s = "{0} grompp -f {1} -c {2}.{3}gro -r {2}.{3}gro -p topol.top -o {2}.{4}.tpr"\
            .format(self.gmx_path, mdpfile, self.basename, inputname, outname)
        return self.execute([s])

    def convert_production(self, mdname, pbc_code, pdb_code):
        s = "{0} trjconv -s {1}.{2}.tpr -f {1}.{2}.xtc -o {1}.{2}.pbc1.xtc -pbc nojump -ur compact"\
            .format(self.gmx_path, self.basename, mdname)
        s2 = "{0} trjconv -s {1}.{2}.tpr -f {1}.{2}.pbc1.xtc -o {1}.{2}.noPBC.xtc -pbc mol -ur compact"\
            .format(self.gmx_path, self.basename, mdname)
        s3 = "{0} trjconv -s {1}.{2}.tpr -f {1}.{2}.noPBC.xtc -o {1}.{2}.pdb"\
            .format(self.gmx_path, self.basename, mdname)
        pipe_codes = [pbc_code, pbc_code, pdb_code]
        self.execute([s, s2, s3], pipe_codes=pipe_codes)

    def rmsd_rmsf(self, mdname, rmsdcode="4 4\n", rmsfcode="3\n"):
        s = "{0} rms -s {1}.{2}.tpr -f {1}.{2}.noPBC.xtc -o rmsd_{1}.{2}.xvg"\
            .format(self.gmx_path, self.basename, mdname)
        s2 = "{0} rmsf -s {1}.{2}.tpr -f {1}.{2}.noPBC.xtc -o rmsf_{1}.{2}.xvg"\
            .format(self.gmx_path, self.basename, mdname)
        pipe_codes = [rmsdcode, rmsfcode]
        self.execute([s, s2], pipe_codes=pipe_codes)

    def energy(self, mdname, energycode="10\n0\n", prefix="potential_"):
        s = "{0} energy -s {1}.{2}.tpr -f {1}.{2}.edr -o {3}{1}.{2}.xvg".format(self.gmx_path, self.basename,
                                                                                mdname, prefix)
        self.execute([s], pipe_codes=[energycode])

    def get_n_atoms_solvated(self):
        filename = "{0}/{1}.solv.ions.gro".format(self.basedir, self.basename)
        with open(filename) as f:
            lines = f.readlines()
        n_atoms = int(lines[1])
        ratio = n_atoms/len(lines)
        if not (ratio > 0.99 and ratio < 1.01):
            raise ValueError("For file {0}, number of atoms was {1} and number of lines was {2} (ratio {3}).".format(
                self.basename, n_atoms, len(lines), ratio
            ))
        return n_atoms


def write_pipefile(workdir, pc, filename):
    with open("{0}/{1}".format(workdir, filename), "w") as f:
        f.write(pc)


def write_sh(workdir, l, filename):
    with open("{0}/{1}".format(workdir, filename), "w") as f:
        f.write(" ".join(l))
        f.write("\n")


def correct_opt_arg(opt_arg):
    if len(opt_arg) == 0:
        return ""
    elif opt_arg[0] != ' ':
        return ' ' + opt_arg
    else:
        return opt_arg


def correct_folder(folder):
    if folder[-1] == "/":
        return folder[:-1]
    return folder
