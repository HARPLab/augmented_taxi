import analyze_sim_data as asd











if __name__ == '__main__':

    path = 'models/augmented_taxi2/N_diagnostic_tests/sim_study'

    filename = '03_02_sim_study_test_final_2_study_200_run_1130.pickle'

    asd.plot_pf_dist(path, filename)