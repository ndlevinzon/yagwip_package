from src.gromacs_sim import GromacsSim, correct_folder
import os
import glob

class Remd:
    """
    Class to set up and initialize a Replica Exchange Molecular Dynamics (REMD) simulation.

    Each replica is simulated at a different temperature, and all simulations follow the
    same preparation pipeline (solvation, minimization, equilibration).
    """
    def __init__(self, base_name, main_dir, base_dirname, temperature_list, basefiles_dir, gmx_path, run_func=None):
        """
                Initialize the REMD setup.

                Parameters:
                - base_name: Base name for simulation files (e.g., "protein")
                - main_dir: Parent directory containing all replicas
                - base_dirname: Subdirectory name template for each replica (e.g., "replica")
                - temperature_list: List of temperatures (in K) for each replica
                - basefiles_dir: Path to MDP and input files to copy into each replica directory
                - gmx_path: Path to the GROMACS executable (e.g., "gmx")
                - run_func: Function to run the preprocessing steps on each replica (default provided)
                """
        if run_func is None:
            run_func = solvate_minimize_equilibrate     # Use default preparation steps if not specified

        # Store configuration
        self.base_name = base_name
        self.main_dir = correct_folder(main_dir)
        self.base_dirname = correct_folder(base_dirname)
        self.temperatures = temperature_list
        self.basefiles_dir = correct_folder(basefiles_dir)
        self.gmx_path = gmx_path
        self.sims = []
        self.make_sims()

        # Create simulation objects and run preparation steps
        for sim in self.sims:
            run_func(sim)

        # Reminder: To launch REMD, run something like:
        # mpirun gmx_mpi mdrun -multidir vp35_0 vp35_1 vp35_2 vp35_3 vp35_4 -deffnm vp35.md1ns -replex 500

    def make_sims(self):
        """
        Create a GromacsSim object for each temperature, and initialize its directory.
        """
        for i, temp in enumerate(self.temperatures):
            simdir = self.main_dir + "/" + self.base_dirname + "_{0}".format(i)
            if not os.path.isdir(simdir):
                os.mkdir(simdir)
            # Instantiate GromacsSim with the directory, base name, and file paths
            self.sims.append(GromacsSim(simdir, self.base_name, self.basefiles_dir, self.gmx_path))
            self.copy_basefiles(simdir, temp)

    def copy_basefiles(self, simdir, temp):
        """
        Copy .mdp and other necessary files into the replica's directory.
        Automatically replaces the temperature in .mdp files.

        Parameters:
        - simdir: Target replica directory
        - temp: Temperature value for this replica
        """
        basefiles = glob.glob("{0}/*".format(self.basefiles_dir))
        for basefile in basefiles:
            filename = basefile.split('/')[-1]
            with open(basefile) as f:
                lines = f.readlines()
            with open(simdir + "/" + filename, "w") as f:
                # Inject replica-specific temperature into ref_t lines
                if filename[-4:] == ".mdp":
                    for line in lines:
                        if line[:5] == "ref_t":
                            f.write("ref_t = {0} {0}\n".format(temp))
                        else:
                            f.write(line)
                else:
                    # For all other files, copy contents as-is
                    for line in lines:
                        f.write(line)


def solvate_minimize_equilibrate(sim, ffcode='15\n', solcode='13\n'):
    """
    Default preparation function for a single simulation.

    This includes:
    - pdb2gmx (force field assignment)
    - solvation
    - ion replacement
    - energy minimization
    - NVT and NPT equilibration
    - TPR generation for production MD

    Parameters:
    - sim: GromacsSim object
    - ffcode: Force field code for stdin
    - solcode: Solvent group code for genion
    """
    sim.pdb2gmx(ffcode)
    sim.solvate()
    sim.genion(solcode, grompp_options="-maxwarn 1")
    sim.em()
    sim.nvt()
    sim.npt()
    sim.prepare_run()






