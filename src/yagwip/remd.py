from yagwip.gromacs_sim import GromacsSim, correct_folder
import os
import glob

class Remd:
    """ Replica exchange MD simulation.
    """
    def __init__(self, base_name, main_dir, base_dirname, temperature_list, basefiles_dir, gmx_path, run_func=None):
        if run_func is None:
            run_func = solvate_minimize_equilibrate
        self.base_name = base_name
        self.main_dir = correct_folder(main_dir)
        self.base_dirname = correct_folder(base_dirname)
        self.temperatures = temperature_list
        self.basefiles_dir = correct_folder(basefiles_dir)
        self.gmx_path = gmx_path
        self.sims = []
        self.make_sims()
        for sim in self.sims:
            run_func(sim)
        # mpirun gmx_mpi mdrun -multidir vp35_0 vp35_1 vp35_2 vp35_3 vp35_4 -deffnm vp35.md1ns -replex 500

    def make_sims(self):
        for i, temp in enumerate(self.temperatures):
            simdir = self.main_dir + "/" + self.base_dirname + "_{0}".format(i)
            if not os.path.isdir(simdir):
                os.mkdir(simdir)
            self.sims.append(GromacsSim(simdir, self.base_name, self.basefiles_dir, self.gmx_path))
            self.copy_basefiles(simdir, temp)

    def copy_basefiles(self, simdir, temp):
        basefiles = glob.glob("{0}/*".format(self.basefiles_dir))
        for basefile in basefiles:
            filename = basefile.split('/')[-1]
            with open(basefile) as f:
                lines = f.readlines()
            with open(simdir + "/" + filename, "w") as f:
                if filename[-4:] == ".mdp":
                    for line in lines:
                        if line[:5] == "ref_t":
                            f.write("ref_t = {0} {0}\n".format(temp))
                        else:
                            f.write(line)
                else:
                    for line in lines:
                        f.write(line)


def solvate_minimize_equilibrate(sim, ffcode='15\n', solcode='13\n'):
    sim.pdb2gmx(ffcode)
    sim.solvate()
    sim.genion(solcode, grompp_options="-maxwarn 1")
    sim.em()
    sim.nvt()
    sim.npt()
    sim.prepare_run()






