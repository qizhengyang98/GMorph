from benchmark_toy import toy_origin, toy_half_conv, toy_all_conv, toy_best_LC_norule, toy_best_LC_rule, toy_best_LC_rule_t002, toy_SA_t0, toy_best_LC_norule_t0, toy_best_LC_rule_t0
from benchmark_scene import scene_origin, scene_SA, scene_best_LC_norule, scene_best_LC_rule, scene_SA_t0, scene_SA_t002, scene_best_LC_norule_t002, scene_best_LC_rule_t002, scene_half_conv, scene_all_conv
from benchmark_face16 import face16_origin, face16_all_conv, face16_SA, face16_best_LC_rule, face16_best_LC_rule_t002, face16_best_LC_rule_t0, face16_SA_002
from benchmark_face import face_origin, face_SA, face_best_LC_norule, face_best_LC_rule, face_half_conv, face_all_conv, face_SA_t002, face_best_LC_norule_t0, face_best_LC_rule_t0, face_best_LC_norule_t002

# acc drop thres = 0.01
results_toymodel = {'origin':toy_origin().eval(), 'merge_half_conv':toy_half_conv().eval(), 'merge_all_conv':toy_all_conv().eval(), 
                    'SA':toy_best_LC_norule().eval(), 'LC_wo_rule':toy_best_LC_norule().eval(), 'LC_w_rule':toy_best_LC_rule().eval()}

results_scene = {'origin':scene_origin().eval(), 'merge_half_conv':scene_half_conv().eval(), 'merge_all_conv':scene_all_conv().eval(),
                 'SA':scene_SA().eval(), 'LC_wo_rule':scene_best_LC_norule().eval(), 'LC_w_rule':scene_best_LC_rule().eval()}

results_face16 = {'origin':face16_origin().eval(),  'merge_half_conv':face16_best_LC_rule().eval(), 'merge_all_conv':face16_all_conv().eval(),
                  'SA':face16_SA().eval(), 'LC_wo_rule':face16_best_LC_rule().eval(), 'LC_w_rule':face16_best_LC_rule().eval()}

results_face = {'origin':face_origin().eval(), 'merge_half_conv':face_half_conv().eval(), 'merge_all_conv':face_all_conv().eval(), 
                'SA':face_SA().eval(), 'LC_wo_rule':face_best_LC_norule().eval(), 'LC_w_rule':face_best_LC_rule().eval()}

# acc drop thres = 0.02
results_toymodel_t002 = {'origin':toy_origin().eval(), 'merge_half_conv':toy_half_conv().eval(), 'merge_all_conv':toy_all_conv().eval(), 
                    'SA':toy_best_LC_norule().eval(), 'LC_wo_rule':toy_best_LC_norule().eval(), 'LC_w_rule':toy_best_LC_rule_t002().eval()}

results_scene_t002 = {'origin':scene_origin().eval(), 'merge_half_conv':scene_half_conv().eval(), 'merge_all_conv':scene_all_conv().eval(),
                      'SA':scene_SA_t002().eval(), 'LC_wo_rule':scene_best_LC_norule_t002().eval(), 'LC_w_rule':scene_best_LC_rule_t002().eval()}

results_face16_t002 = {'origin':face16_origin().eval(),  'merge_half_conv':face16_best_LC_rule().eval(), 'merge_all_conv':face16_all_conv().eval(),
                  'SA':face16_SA_002().eval(), 'LC_wo_rule':face16_best_LC_rule_t002().eval(), 'LC_w_rule':face16_best_LC_rule_t002().eval()}

results_face_t002 = {'origin':face_origin().eval(), 'merge_half_conv':face_half_conv().eval(), 'merge_all_conv':face_all_conv().eval(), 
                     'SA':face_best_LC_norule_t002().eval(), 'LC_wo_rule':face_best_LC_norule_t002().eval(), 'LC_w_rule':face_SA_t002().eval()}

# acc drop thres = 0
results_toymodel_t0 = {'origin':toy_origin().eval(), 'merge_half_conv':toy_half_conv().eval(), 'merge_all_conv':toy_all_conv().eval(), 
                    'SA':toy_SA_t0().eval(), 'LC_wo_rule':toy_best_LC_norule_t0().eval(), 'LC_w_rule':toy_best_LC_rule_t0().eval()}

results_scene_t0 = {'origin':scene_origin().eval(), 'merge_half_conv':scene_half_conv().eval(), 'merge_all_conv':scene_all_conv().eval(),
                    'SA':scene_SA_t0().eval(), 'LC_wo_rule':scene_SA_t0().eval(), 'LC_w_rule':scene_origin().eval()}

results_face16_t0 = {'origin':face16_origin().eval(), 'merge_half_conv':face16_best_LC_rule().eval(), 'merge_all_conv':face16_all_conv().eval(),
                  'SA':None, 'LC_wo_rule':face16_best_LC_rule_t0().eval(), 'LC_w_rule':face16_best_LC_rule_t0().eval()}

results_face_t0 = {'origin':face_origin().eval(), 'merge_half_conv':face_half_conv().eval(), 'merge_all_conv':face_all_conv().eval(), 
                   'SA':face_best_LC_norule_t0().eval(), 'LC_wo_rule':face_best_LC_norule_t0().eval(), 'LC_w_rule':face_best_LC_rule_t0().eval()}