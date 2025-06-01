import subprocess
import numpy as np
import math


def run_gromacs_command(command, pipe_input=None, debug=False, logger=None):
    if logger:
        logger.info(f"[RUNNING] {command}")
    else:
        print(f"[RUNNING] {command}")

    if debug:
        if logger:
            logger.debug("[DEBUG MODE] Command not executed.")
        else:
            print("[DEBUG MODE] Command not executed.")
        return

    try:
        result = subprocess.run(
            command,
            input=pipe_input,
            shell=True,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if logger:
                logger.error(f"Command failed with return code {result.returncode}")
                logger.error(result.stderr.strip())
                logger.info(result.stdout.strip())
            else:
                print(f"[ERROR] Command failed with return code {result.returncode}")
                print("[STDERR]", result.stderr.strip())
                print("[STDOUT]", result.stdout.strip())
        else:
            if logger:
                logger.info(result.stdout.strip())
            else:
                print(result.stdout.strip())

    except Exception as e:
        if logger:
            logger.exception(f"Failed to run command: {e}")
        else:
            print(f"[EXCEPTION] Failed to run command: {e}")


# Based on http://dx.doi.org/10.1039/b716554d
def tremd_temperature_ladder(Nw, Np, Tlow, Thigh, Pdes, WC=3, PC=1, Hff=0, Vs=0, Alg=0, Tol=0.001):
    # Constants
    A0, A1 = -59.2194, 0.07594
    B0, B1 = -22.8396, 0.01347
    D0, D1 = 1.1677, 0.002976
    kB = 0.008314
    maxiter = 100

    if Hff == 0:
        Nh = round(Np * 0.5134)
        VC = round(1.91 * Nh) if Vs == 1 else 0
        Nprot = Np
    else:
        Npp = round(Np / 0.65957)
        Nh = round(Np * 0.22)
        VC = round(Np + 1.91 * Nh) if Vs == 1 else 0
        Nprot = Npp

    NC = Nh if PC == 1 else Np if PC == 2 else 0
    Ndf = (9 - WC) * Nw + 3 * Np - NC - VC
    FlexEner = 0.5 * kB * (NC + VC + WC * Nw)

    def myeval(m12, s12, CC, u):
        arg = -CC * u - (u - m12) ** 2 / (2 * s12 ** 2)
        return np.exp(arg)

    def myintegral(m12, s12, CC):
        umax = m12 + 5 * s12
        du = umax / 100
        u_vals = np.arange(0, umax, du)
        vals = [myeval(m12, s12, CC, u + du / 2) for u in u_vals]
        pi = np.pi
        return du * sum(vals) / (s12 * np.sqrt(2 * pi))

    temps = [Tlow]
    while temps[-1] < Thigh:
        T1 = temps[-1]
        T2 = T1 + 1 if T1 + 1 < Thigh else Thigh
        low, high = T1, Thigh
        iter_count = 0
        piter = 0
        forward = True

        while abs(Pdes - piter) > Tol and iter_count < maxiter:
            iter_count += 1
            mu12 = (T2 - T1) * ((A1 * Nw) + (B1 * Nprot) - FlexEner)
            CC = (1 / kB) * ((1 / T1) - (1 / T2))
            var = Ndf * (D1 ** 2 * (T1 ** 2 + T2 ** 2) + 2 * D1 * D0 * (T1 + T2) + 2 * D0 ** 2)
            sig12 = np.sqrt(var)

            I1 = 0.5 * math.erfc(mu12 / (sig12 * np.sqrt(2)))
            I2 = myintegral(mu12, sig12, CC)
            piter = I1 + I2

            if piter > Pdes:
                if forward:
                    T2 += 1.0
                else:
                    low = T2
                    T2 = low + (high - low) / 2
                if T2 >= Thigh:
                    T2 = Thigh
            else:
                if forward:
                    forward = False
                    low = T2 - 1.0
                high = T2
                T2 = low + (high - low) / 2

        temps.append(round(T2, 2))

    print("Please Cite: Alexandra Patriksson and David van der Spoel, A temperature predictor for parallel tempering \n"
          "simulations Phys. Chem. Chem. Phys., 10 pp. 2073-2077 (2008)")

    return temps