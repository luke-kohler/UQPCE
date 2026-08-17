import numpy as np
from scipy.optimize import minimize

import openmdao.api as om


from disciplines.BreguetRange import BreguetRangeComp
from disciplines.aero import AeroComp
from disciplines.total_mass_comp import TotalMassComp
from disciplines.propulsion import PropulsionComp
from disciplines.weight import EngineWeightComp, WeightsComp
from disciplines.doc import DOC
from disciplines.dpm import Dpm

from fixed import *
#from .helpers import *
from .organize import CoupledDisciplines
from .organize import configure_subsystems


TUNING_NAMES = [
    'p_base',
    'kv_base',
    'ks_base',
    'kw_base',
    'eta_base',
    'alpha_base',
    'beta_base',
    'C_time'
]

def initialize(prob, params=parameters):
    prob.set_val('V_cruise', params['V_cruise'])
    prob.set_val('S', params['S'])
    prob.set_val('AR', params['AR'])
    prob.set_val('SFC_tech', params['SFC_tech'])

    # Tuning Parameters
    prob.set_val('e_base', parameters['e_oswald_base'])
    prob.set_val('C_D0_base', parameters['CD0_base'])
    prob.set_val('Cf_base', parameters['Cf_base'])
    prob.set_val('fsys_base', tuning['fsys_base'])          # fraction of total mass comprising 'systems' and stuff
    prob.set_val('kw_base', tuning['kw_base'])              # wing weight regression/fit tuning parameter
    prob.set_val('p_base', tuning['p_base'])                # off (faster) design velocity wing weight penalty exponent parameter
    prob.set_val('eta_base', tuning['eta_base'])            # tuning paramter to change effect SFC_tech has on changing SFC_ref
    prob.set_val('kv_base', tuning['kv_base'])              # off design veloicty penalty to increase SFC qudratically about V_ref
    prob.set_val('beta_base', tuning['beta_base'])          # strength of increase/decrease of amortized engine cost due to SFC_tech
    prob.set_val('alpha_base', tuning['alpha_base'])        # strength of increase/decrease of engine mass due to SFC_tech
    prob.set_val('ks_base', tuning['ks_base'])     

def build_inner_problem():
    prob = om.Problem(reports=False)
    configure_subsystems(prob)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 100
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = False

    # Inner design optimization: aircraft variables
    prob.model.add_design_var(
        'S',
        lower=100.0,
        upper=180.0,
        ref0=100,
        ref=180
    )

    prob.model.add_design_var(
        'AR',
        lower=5.0,
        upper=20.0,
        ref0=5.0,
        ref=20
    )

    prob.model.add_design_var(
        'V_cruise',
        lower=220.0,
        upper=260.0,
        ref0=220.0,
        ref=260
    )

    prob.model.add_design_var(
        'SFC_tech',
        lower=-1.0,
        upper=1.0,
        ref0=-1.0,
        ref=1.0
    )

    # Inner objective: normal aircraft optimization
    prob.model.add_objective(
        'DOC',
        ref=1.0e4
    )

    prob.model.add_constraint(
        'm_fuel',
        lower=1000.0,
        upper=70000.0,
        ref0=10000.0,
        ref=70000
    )

    #prob.model.add_constraint(
    #    'CL',
    #    lower=0.4,
    #    upper=0.65,
    #    ref0=0.4,
    #    ref=0.65
    #)

    prob.model.add_constraint(
            'WL',
            lower=6000,
            upper=6400,
            ref0=6000,
            ref=6400
    )

    prob.setup()
    initialize(prob)

    return prob


def calibration_objective(x):
    prob = build_inner_problem()

    # x = [p_base, kv_base, ks_base]
    for name, value in zip(TUNING_NAMES, x):
        prob.set_val(name, value)

    try:
        prob.run_driver()
    except Exception:
        return 1.0e12

    S = prob.get_val('S').item()
    AR = prob.get_val('AR').item()
    V = prob.get_val('V_cruise').item()
    SFC_tech = prob.get_val('SFC_tech').item()
    CL = prob.get_val('CL').item()
    m_fuel = prob.get_val('m_fuel').item()

    AR_target = 11.0
    S_target = 133.0
    V_target = 244.0
    SFC_tech_target = 0.24 
    CL_target = 0.61

    J = (
        ((AR - AR_target) / AR_target) ** 2
        + ((S - S_target) / S_target) ** 2
        + ((V - V_target) / V_target) ** 2
        + ((SFC_tech - SFC_tech_target) / SFC_tech_target) ** 2
        #((CL - CL_target) / CL_target) **2
    ) 

    print(
        f"J={J:.6e}, "
        f"S={S:.3f}, AR={AR:.3f}, V={V:.3f}, "
        f"SFC_tech={SFC_tech:.3f}, "
        f"p={x[0]:.6g}, kv={x[1]:.6g}, ks={x[2]:.6g}"
    )

    print("CL:",CL)
    print("m_fuel:",m_fuel)
    print("R",prob.get_val('R'))

    return J


def main():
    x0 = np.array([
        tuning['p_base'],
        tuning['kv_base'],
        tuning['ks_base'],
        tuning['kw_base'],
        tuning['eta_base'],
        tuning['alpha_base'],
        tuning['beta_base'],
        parameters['C_time']
    ])

    bounds = [
        (1.0, 9.0),       # p_base
        (200, 700.0),     # kv_base
        (1.0e-6, 8e-4),   # ks_base
        (1,100),
        (0.1,1.0),
        (0.1,1.0),
        (0.1,1.0),
        (0.3,0.75)
    ]

    result = minimize(
        calibration_objective,
        x0,
        method='Powell',
        bounds=bounds,
        options={
            'maxiter': 500,
            'disp': True,
        }
    )

    print("\nBest tuning parameters:")
    for name, value in zip(TUNING_NAMES, result.x):
        print(name, value)

    print("\nFinal calibration objective:", result.fun)



if __name__ == "__main__":
    main()