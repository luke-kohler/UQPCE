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
from .organize import CoupledDisciplines, configure_subsystems
from fixed import *


#steps
#add subssystem whose outputs are the optima of the determ opt
#have thiose feed into second component whose output is the objective 
#figure out p_base, k_vbase, and ks_base such that design vars are in 
# a certain region type shit

class Deterministic(om.Group):

    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']

        self.add_subsystem(
        'Prop', 
        PropulsionComp(vec_size=n),
        promotes_inputs=['V_cruise', 'SFC_tech',
                         'SFC_ref', 'V_ref',
                         'eta_base', 'kv_base',
                         'delta_eta', 'delta_kv'],
        promotes_outputs=['SFC']
        )

        # Engine Weight Component
        self.add_subsystem(
            'Engine', 
            EngineWeightComp(vec_size=n), 
            promotes_inputs=['SFC_tech', 
                            'm_eng_ref', 'alpha_base',
                            'delta_alpha'],
            promotes_outputs=['m_engine']
        )

        self.add_subsystem(
            'AeroStruct', 
            CoupledDisciplines(vec_size=n), 
            promotes_inputs=['V_cruise', 'S', 'AR',
                            'C_D0_base', 'ks_base', 'e_base', 'S_0',
                            'kw_base', 'fsys_base', 'p_base',
                            'V_ref', 'R_target', 'SFC',
                            'm_fuse', 'm_payload', 'm_engine',
                            'delta_CD0', 'delta_ks', 'delta_e',
                            'delta_fsys', 'delta_kw', 'delta_p'], 
            promotes_outputs=['m_fuel', 'm_empty', 'm_wing',
                            'm_total', 'LD', 'CL', 'CD', 'WL', 'R']
        )

        self.add_subsystem(
            'DOC_objective', 
            DOC(vec_size=n), 
            promotes_inputs=['V_cruise', 'SFC_tech',
                            'Cf_base', 'beta_base',
                            'C_time', 'k_acq', 'C_eng_ref', 
                            'delta_beta', 'delta_Cf', 
                            'R', 'm_fuel'], 
            promotes_outputs=['DOC']
        )

def create_inner_problem():
    prob = om.Problem()

    # Explicit sources for the INNER optimization design variables
    design_vars = om.IndepVarComp()

    design_vars.add_output('S', val=124.58, units='m**2')
    design_vars.add_output('AR', val=9.45)
    design_vars.add_output('V_cruise', val=240.5, units='m/s')
    design_vars.add_output('SFC_tech', val=0.0)

    prob.model.add_subsystem(
        'design_vars',
        design_vars,
        promotes_outputs=[
            'S',
            'AR',
            'V_cruise',
            'SFC_tech',
        ],
    )
    
    prob.model.add_subsystem("Inner_Problem",Deterministic(),
                                promotes_inputs=['*'],
                                promotes_outputs=['*'])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 1000
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True
    
    # Declare Design variables
    prob.model.add_design_var('S', lower=100.0, upper=300.0, ref=124.6)
    prob.model.add_design_var('AR', lower=3.0, upper=100.0, ref=9.45)
    prob.model.add_design_var('V_cruise', lower=100, upper=300, ref=1)
    prob.model.add_design_var('SFC_tech', lower=-1, upper=1, ref=1)
    
    # Declare Objective Function
    prob.model.add_objective('DOC', ref=1.0e4)

    return prob

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
    #prob.set_val('p_base', tuning['p_base'])                # off (faster) design velocity wing weight penalty exponent parameter
    prob.set_val('eta_base', tuning['eta_base'])            # tuning paramter to change effect SFC_tech has on changing SFC_ref
    #prob.set_val('kv_base', tuning['kv_base'])              # off design veloicty penalty to increase SFC qudratically about V_ref
    prob.set_val('beta_base', tuning['beta_base'])          # strength of increase/decrease of amortized engine cost due to SFC_tech
    prob.set_val('alpha_base', tuning['alpha_base'])        # strength of increase/decrease of engine mass due to SFC_tech
    #prob.set_val('ks_base', tuning['ks_base'])    

class ResettingSubmodelComp(om.SubmodelComp):

    def compute(self, inputs, outputs):

        p = self.problem

        # Always start the inner optimization from the same design
        p.set_val('S', parameters['S'])
        p.set_val('AR', parameters['AR'])
        p.set_val('V_cruise', parameters['V_cruise'])
        p.set_val('SFC_tech', parameters['SFC_tech'])

        # Reset nonlinear state too
        p.set_val('m_fuel', 20000.0, units='kg')

        super().compute(inputs, outputs)

class TuningObjective(om.ExplicitComponent):

    def setup(self):

        self.add_input('S_opt')
        self.add_input('AR_opt')
        self.add_input('V_opt')
        self.add_input('SFC_tech_opt')
        self.add_input('CL_opt')

        self.add_output('tuning_objective')

    def compute(self, inputs, outputs):

        S = inputs['S_opt']
        AR = inputs['AR_opt']
        V = inputs['V_opt']
        tech = inputs['SFC_tech_opt']
        CL = inputs['CL_opt']

        # temporary example
        J = (
            ((S - 140.58) )**2
            + ((AR - 9.45) )**2
            + ((V - 240.5) )**2
        )

        outputs['tuning_objective'] = J


def main():

    outer_problem = om.Problem()    

    inner_problem = create_inner_problem()

    inner_problem_component = ResettingSubmodelComp(
        problem=inner_problem,
        inputs=['p_base','kv_base','ks_base'],
        outputs=[
            ('S', 'S_opt'),
            ('AR', 'AR_opt'),
            ('V_cruise', 'V_opt'),
            ('SFC_tech', 'SFC_tech_opt'),
            ('CL', 'CL_opt'),
            ('DOC', 'DOC_opt'),
        ],

        reports=False,
    )

    outer_problem.model.add_subsystem(
        'InnerOptimization',
        inner_problem_component,
        promotes_inputs=[
            'p_base',
            'kv_base',
            'ks_base',
        ],
        promotes_outputs=[
            'S_opt',
            'AR_opt',
            'V_opt',
            'SFC_tech_opt',
            'CL_opt',
            'DOC_opt',
        ],
    )

    outer_problem.model.add_subsystem(
        'TuningObjective',
        TuningObjective(),
        promotes_inputs=[
            'S_opt',
            'AR_opt',
            'V_opt',
            'SFC_tech_opt',
            'CL_opt',
        ],
        promotes_outputs=['tuning_objective'],
    )

    # OUTER design variables

    outer_problem.driver = om.ScipyOptimizeDriver()
    outer_problem.driver.options['optimizer'] = 'SLSQP'
    outer_problem.driver.options['maxiter'] = 100
    outer_problem.driver.options['tol'] = 1e-4
    outer_problem.driver.options['disp'] = True

    outer_problem.model.add_design_var(
        'p_base',
        lower = 2.0,
        upper = 8.0,
        ref0=2.0,
        ref=8.0
    )

    outer_problem.model.add_design_var(
        'kv_base',
        lower = 10,
        upper = 1000,
        ref0=10.0,
        ref=1000.0
    )

    outer_problem.model.add_design_var(
        'ks_base',
        lower = 0.0,
        upper = 5e-3,
        ref0=0.0,
        ref=5e-3
    )

    outer_problem.model.add_objective('tuning_objective')
    outer_problem.model.approx_totals(
    method='fd',
    form='forward',
    step=1e-2,
    step_calc='rel_avg',

    )
    outer_problem.setup()

    outer_problem.set_val('p_base', tuning['p_base'])
    outer_problem.set_val('kv_base', tuning['kv_base'])
    outer_problem.set_val('ks_base', tuning['ks_base'])

    # Set fixed parameters here, or preferably inside
    # the deterministic model using set_input_defaults.
    initialize(inner_problem)
    outer_problem.run_driver()


    print("Tuning params\n")

    print(outer_problem.get_val('p_base'))
    print(outer_problem.get_val('kv_base'))
    print(outer_problem.get_val('ks_base'))



     








if __name__ == "__main__":
    main()
