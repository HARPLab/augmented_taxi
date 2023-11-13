# def downsample_team_models_pf(particles_team, n_models):

    
#     sampled_team_models = {}
#     sampled_team_model_weights = {}

#     for member_id in particles_team:
#         sampled_human_model_idxs = []
#         particles_team[member_id].cluster()
#         if len(particles_team[member_id].cluster_centers) > n_models:
#             # if there are more clusters than number of sought human models, return the spherical centroids of the top n
#             # most frequently counted cluster indexes selected by systematic resampling
#             while len(sampled_human_model_idxs) < n_models:
#                 indexes = p_utils.systematic_resample(particles_team[member_id].cluster_weights)
#                 unique_idxs, counts = np.unique(indexes, return_counts=True)
#                 # order the unique indexes via their frequency
#                 unique_idxs_sorted = [x for _, x in sorted(zip(counts, unique_idxs), reverse=True)]

#                 for idx in unique_idxs_sorted:
#                     # add new unique indexes to the human models that will be considered for counterfactual reasoning
#                     if idx not in sampled_human_model_idxs:
#                         sampled_human_model_idxs.append(idx)

#                     if len(sampled_human_model_idxs) == n_models:
#                         break

#             sampled_human_models = [particles_team[member_id].cluster_centers[i] for i in sampled_human_model_idxs]
#             sampled_human_model_weights = np.array([particles_team[member_id].cluster_weights[i] for i in sampled_human_model_idxs])
#             sampled_human_model_weights /= np.sum(sampled_human_model_weights)  # normalize
#         elif len(particles_team[member_id].cluster_centers) == n_models:
#             sampled_human_models = particles_team[member_id].cluster_centers
#             sampled_human_model_weights = np.array(particles_team[member_id].cluster_weights) # should already be normalized
#         else:
#             # if there are fewer clusters than number of sought human models, use systematic sampling to determine how many
#             # particles from each cluster to return (using the k-cities algorithm to ensure that they are diverse)
#             indexes = p_utils.systematic_resample(particles_team[member_id].cluster_weights, N=n_models)
#             unique_idxs, counts = np.unique(indexes, return_counts=True)

#             sampled_human_models = []
#             sampled_human_model_weights = []
#             for j, unique_idx in enumerate(unique_idxs):
#                 # particles of this cluster
#                 clustered_particles = particles_team[member_id].positions[np.where(particles_team[member_id].cluster_assignments == unique_idx)]
#                 clustered_particles_weights = particles_team[member_id].weights[np.where(particles_team[member_id].cluster_assignments == unique_idx)]

#                 # use the k-cities algorithm to obtain a diverse sample of weights from this cluster
#                 pairwise = metrics.pairwise.euclidean_distances(clustered_particles.reshape(-1, 3))
#                 select_idxs = BEC_helpers.selectKcities(pairwise.shape[0], pairwise, counts[j])
#                 sampled_human_models.extend(clustered_particles[select_idxs])
#                 sampled_human_model_weights.extend(clustered_particles_weights[select_idxs])

#             sampled_human_model_weights = np.array(sampled_human_model_weights)
#             sampled_human_model_weights /= np.sum(sampled_human_model_weights)

#         # update member models
#         sampled_team_models[member_id] = sampled_human_models
#         sampled_team_model_weights[member_id] = sampled_human_model_weights
    

#     return sampled_team_models, sampled_team_model_weights


def sample_human_models_uniform_joint_knowledge(joint_constraints, n_models):
    '''
    Summary: sample representative weights that the human could currently attribute to the agent, by greedily selecting
    points that minimize the maximize distance to any other point (k-centers problem)
    '''

    sample_human_models = []

    if len(joint_constraints) > 0:
        joint_constraints_matrix = np.vstack(joint_constraints)

        # obtain x, y, z coordinates on the sphere that obey the constraints
        valid_sph_x, valid_sph_y, valid_sph_z = sample_valid_region_jk(joint_constraints_matrix, 0, 2 * np.pi, 0, np.pi, 1000, 1000)

        if len(valid_sph_x) == 0:
            print(colored("Was unable to sample valid human models within the BEC (which is likely too small).",
                        'red'))
            return sample_human_models

        # resample coordinates on the sphere within the valid region (for higher density)
        sph_polygon = cg.cart2sph(np.array([valid_sph_x, valid_sph_y, valid_sph_z]).T)
        sph_polygon_ele = sph_polygon[:, 0]
        sph_polygon_azi = sph_polygon[:, 1]

        min_azi = min(sph_polygon_azi)
        max_azi = max(sph_polygon_azi)
        min_ele = min(sph_polygon_ele)
        max_ele = max(sph_polygon_ele)

        # sample according to the inverse CDF of the uniform distribution along the sphere
        u_low = min_azi / (2 * np.pi)
        u_high = max_azi / (2 * np.pi)
        v_low = (1 - np.cos(min_ele)) / 2
        v_high = (1 - np.cos(max_ele)) / 2

        n_discrete_samples = 100
        while len(sample_human_models) < n_models:
            n_discrete_samples += 20
            theta = 2 * np.pi * np.linspace(u_low, u_high, n_discrete_samples)
            phi = np.arccos(1 - 2 * np.linspace(v_low, v_high, n_discrete_samples))

            # reject points that fall outside of the desired area

            # see which points on the sphere obey atleast one of constraints. 
            # see which points on the sphere obey atleast one of the original constraints
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            sph_points = np.array(cg.sph2cart(np.array([phi_grid.flatten(), theta_grid.flatten()]).T))
            dist_to_plane = joint_constraints_matrix.dot(sph_points.T)
            n_joint_constraints_satisfied = np.sum(dist_to_plane >= 0, axis=0)
            # n_min_constraints = constraints_matrix.shape[0]

            idx_valid_sph_points = np.where(n_joint_constraints_satisfied > 0)[0]
            valid_sph_points = sph_points[idx_valid_sph_points, :]

            # greedily select k 'centers' such that the maximum distance from any point to a center is minimized
            # solution is never worse than twice the optimal solution (2-approximation greedy algorithm)
            # https://www.geeksforgeeks.org/k-centers-problem-set-1-greedy-approximate-algorithm/
            if len(valid_sph_points) == n_models:
                sample_human_models.extend(valid_sph_points)
            else:
                valid_sph_points_latllong = cg.cart2latlong(valid_sph_points)
                pairwise = metrics.pairwise.haversine_distances(valid_sph_points_latllong)
                select_idxs = BEC_helpers.selectKcities(pairwise.shape[0], pairwise, n_models)
                select_sph_points = valid_sph_points[select_idxs]
                # reshape so that each element is a valid weight vector
                select_sph_points = select_sph_points.reshape(select_sph_points.shape[0], 1, select_sph_points.shape[1])
                sample_human_models.extend(select_sph_points)
    else:
        points = cg.generate_equidistributed_points_on_sphere(n_models)
        points = np.expand_dims(points, 1)
        sample_human_models.extend(points)

    return sample_human_models


#############################

# calculating knowledge area based on particle filter

    # Methods using particle filters - unreliable; varies based on the number of particles!
    #     # print('min_BEC_constraints: ', [min_BEC_constraints])
    #     min_BEC_area = BEC_helpers.calc_solid_angles([min_BEC_constraints])
    #     # print('knowledge: ', team_knowledge[knowledge_type])
    #     knowledge_area = BEC_helpers.calc_solid_angles([team_knowledge[knowledge_type]])
        
    #     # n_particles = 5000
    #     n_particles = 500
    #     knowledge_particles = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform(team_knowledge[knowledge_type], n_particles))

    #     const_id = []
    #     for j, x in enumerate(knowledge_particles.positions):

    #         all_constraints_satisfied = True
    #         for constraint in min_BEC_constraints:
    #             dot = constraint.dot(x.T)

    #             if dot < 0:
    #                 all_constraints_satisfied = False
            
    #         if all_constraints_satisfied:
    #             const_id.append(j)
        
    #     BEC_overlap_area = min(min_BEC_area, len(const_id)/n_particles * np.array(knowledge_area))
        
    #     # # Method 1: Calculates the knowledge level (0 to 1) based on two factors: ratio of BEC area to knowledge area and % of overlap of BEC area with knowledge area
    #     # knowledge_spread = np.array(knowledge_area)/np.array(min_BEC_area)
    #     # BEC_overlap_ratio = BEC_overlap_area/np.array(min_BEC_area)
    #     # knowledge_level[knowledge_type] = 0.5*BEC_overlap_ratio + 0.5/knowledge_spread

    #     # Method 2: Calculate knowledge level based on the ratio of BEC overlap area with the total knowledge area
    #     knowledge_level[knowledge_type] = BEC_overlap_area/np.array(knowledge_area)

        ##############################################



def majority_rules_non_intersecting_team_constraints(test_constraints_team, weights, step_cost_flag, test_flag = False):

    
    # non_intersecting_flag_cnst = []
    intersecting_constraint_idx = []
    max_intersecting_cnsts = []
    intersecting_cnst = []
    alternate_team_constraints = []
    N_max_cnst = 0

    # N_loops = 1
    # if test_flag == True:
    #     N_loops = len(test_constraints_team)
    # N_max_cnst = np.zeros(N_loops)

    # print(colored('Number of loops:' + str(N_loops), 'blue'))
    print('test_constraints_team: ', test_constraints_team)

    for loop_id in range(1):
        print('Checking for majority constrainst for loop: ', loop_id)
        
        if not test_flag:
            test_constraints_team_expanded = copy.deepcopy(test_constraints_team)
            N_intersecting_sets = len(test_constraints_team_expanded)

        else:
            test_constraints_team_expanded = copy.deepcopy(test_constraints_team)
            N_intersecting_sets = len(test_constraints_team)



        
        while N_intersecting_sets > 1:
            intersections_to_check = list(itertools.combinations(range(len(test_constraints_team_expanded)), N_intersecting_sets))

            print('Constraint list: ', test_constraints_team_expanded)
            print('N_intersecting_sets: ', N_intersecting_sets)
            print('Intersections to check: ', intersections_to_check)

            for i in range(len(intersections_to_check)):
                constraints = []
                for j in intersections_to_check[i]:
                    if test_flag:
                        constraints.extend(test_constraints_team_expanded[j])
                    else:
                        constraints.append(test_constraints_team_expanded[j])

                print('Constraints to check for non-intersection: ', constraints)

                non_intersecting_cnst_flag, _ = check_for_non_intersecting_constraints(constraints, weights, step_cost_flag)
                # non_intersecting_flag_cnst.extend([non_intersecting_cnst_flag, constraints])
                # # print('non_intersecting_cnst_flag: ', non_intersecting_cnst_flag, ' constraints: ', constraints)

                if not non_intersecting_cnst_flag:
                    # # print('max_intersecting_cnsts: ', max_intersecting_cnsts)
                    # # print('constraints: ', constraints)

                    if len(max_intersecting_cnsts) == loop_id:
                        if loop_id == 0:
                            max_intersecting_cnsts = [[copy.deepcopy(constraints)]]
                            intersecting_cnst = [[intersections_to_check[i]]]
                        else:
                            max_intersecting_cnsts.append([copy.deepcopy(constraints)])
                            intersecting_cnst.append([intersections_to_check[i]])
                        N_max_cnst[loop_id] = len(constraints)
                    else:
                        # # print('max_intersecting_cnsts[0]): ', max_intersecting_cnsts[0])
                        if len(constraints) > N_max_cnst[loop_id]:   # update the max intersecting constraints
                            max_intersecting_cnsts[loop_id] = [copy.deepcopy(constraints)]
                            intersecting_cnst[loop_id] = [intersections_to_check[i]]
                            N_max_cnst[loop_id] = len(constraints)
                        elif len(constraints) == N_max_cnst[loop_id]:         # joint max intersecting constraints
                            max_intersecting_cnsts[loop_id].append(constraints)
                            intersecting_cnst[loop_id].append(intersections_to_check[i])
                    # # print('Updated max int cnts: ', max_intersecting_cnsts)
                    
                # # plot
                # if non_intersecting_cnst_flag:
                #     # print('Plotting..')
                #     # plot actual knowledge constraints for this knowledge type
                #     fig = plt.figure()
                #     ax = fig.add_subplot(projection='3d')
                #     utils_teams.visualize_planes_team(constraints, fig=fig, ax=ax, alpha=0.5)
                #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
                #     poly = Polyhedron.Polyhedron(ieqs=ieqs)
                #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, color = 'b')
                #     plt.show()

            N_intersecting_sets -= 1


        print('max_intersecting_cnsts for loop ', loop_id, ': ', max_intersecting_cnsts[loop_id])


        # choose the sets of constraints that are the most intersecting
        # if len(max_intersecting_cnsts) == 1:
        #     if len(alternate_team_constraints) == 0:
        #         alternate_team_constraints = max_intersecting_cnsts[0]
        #         intersecting_constraint_idx = [intersecting_cnst[0]]
        #     else:
        #         alternate_team_constraints.append(max_intersecting_cnsts[0])
        #         intersecting_constraint_idx.append(intersecting_cnst[0])
        # elif len(max_intersecting_cnsts) > 1:
        #     rand_index = random.randint(0, len(max_intersecting_cnsts)-1)
        #     if len(alternate_team_constraints) == 0:
        #         alternate_team_constraints = max_intersecting_cnsts[rand_index]
        #         intersecting_constraint_idx = [intersecting_cnst[rand_index]]
        #     else:
        #         alternate_team_constraints.append(max_intersecting_cnsts[rand_index])
        #         intersecting_constraint_idx.append(intersecting_cnst[rand_index])
        # else:
        #     # raise RuntimeError('No intersecting constraints found!')
        #     alternate_team_constraints = []
        #     intersecting_constraints = []

    print('max_intersecting_cnsts: ', max_intersecting_cnsts) 

    ## max intersecting constraints
    if len(max_intersecting_cnsts) > 0:
        for max_int_cnsts in max_intersecting_cnsts:
            if len(max_int_cnsts) == 1:
                if len(alternate_team_constraints) == 0:
                    alternate_team_constraints = max_int_cnsts[0]
                else:
                    alternate_team_constraints.extend(max_int_cnsts[0])
            elif len(max_int_cnsts) > 1:
                rand_index = random.randint(0, len(max_int_cnsts)-1)
                if len(alternate_team_constraints) == 0:
                    alternate_team_constraints = max_int_cnsts[rand_index]
                else:
                    alternate_team_constraints.extend(max_int_cnsts[rand_index])


    
    # # print(non_intersecting_flag_cnst)
    

    return alternate_team_constraints, intersecting_cnst