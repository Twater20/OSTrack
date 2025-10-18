from sympy import false
import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'
dataset_name = 'lasot_extension_subset'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""
# trackers.extend(trackerlist(name='kmostrack', parameter_name='vitb_256_mae_ce_96x1_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='KMOSTrack256'))
# trackers.extend(trackerlist(name='kmostrack', parameter_name='vitb_256_mae_ce_96x1_ep300-4', dataset_name=dataset_name,
#                             run_ids=None, display_name='KMOSTrack256-4'))
# trackers.extend(trackerlist(name='roistrack', parameter_name='vitb_256_mae_ce_96x1_ep300-3', dataset_name=dataset_name,
#                             run_ids=None, display_name='ROISTrack256-3'))
# trackers.extend(trackerlist(name='roistrack', parameter_name='vitb_256_mae_ce_96x1_ep300-4', dataset_name=dataset_name,
#                             run_ids=None, display_name='ROISTrack256-4'))
# trackers.extend(trackerlist(name='roistrack', parameter_name='vitb_256_mae_ce_96x1_ep300-5', dataset_name=dataset_name,
#                             run_ids=None, display_name='ROISTrack256-5'))
# trackers.extend(trackerlist(name='roistrack', parameter_name='vitb_256_mae_ce_96x1_ep300-6', dataset_name=dataset_name,
#                             run_ids=None, display_name='ROISTrack256-6'))
# trackers.extend(trackerlist(name='roistrack', parameter_name='vitb_256_mae_ce_96x1_ep300-7', dataset_name=dataset_name,
#                             run_ids=None, display_name='ROISTrack256-7'))
# trackers.extend(trackerlist(name='roistrack', parameter_name='vitb_256_mae_ce_96x1_ep300-8', dataset_name=dataset_name,
#                             run_ids=None, display_name='ROISTrack256-8'))
# trackers.extend(trackerlist(name='cvtostrack', parameter_name='cvt13_256_mae_ce_96x1_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='CVTOSTrack256-1'))
# trackers.extend(trackerlist(name='cvtostrack', parameter_name='cvt13_256_mae_ce_96x1_ep300-0', dataset_name=dataset_name,
#                             run_ids=None, display_name='CVTOSTrack256'))
# trackers.extend(trackerlist(name='timostrack', parameter_name='vitb_256_mae_ce_96x1_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='TIMOSTrack256'))
# trackers.extend(trackerlist(name='timostrack', parameter_name='vitb_256_mae_ce_96x1_ep300-10', dataset_name=dataset_name,
#                             run_ids=None, display_name='TIMOSTrack256'))
# trackers.extend(trackerlist(name='timostrack', parameter_name='vitb_256_mae_ce_96x1_ep300-1', dataset_name=dataset_name,
#                             run_ids=None, display_name='TIMOSTrack256-1'))
# trackers.extend(trackerlist(name='timostrack', parameter_name='vitb_256_mae_ce_96x1_ep300-2', dataset_name=dataset_name,
#                             run_ids=None, display_name='TIMOSTrack256-2'))
# trackers.extend(trackerlist(name='timostrack', parameter_name='vitb_256_mae_ce_96x1_ep300-3', dataset_name=dataset_name,
#                             run_ids=None, display_name='TIMOSTrack256-3'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_96x1_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='timostrack', parameter_name='vitb_256_mae_ce_64x2_ep340-lasot-10(teto)', dataset_name=dataset_name,
#                             run_ids=None, display_name='TIMOSTrack256-10'))
trackers.extend(trackerlist(name='timostrack', parameter_name='vitb_256_mae_ce_64x2_ep340', dataset_name=dataset_name,
                            run_ids=None, display_name='TIMOSTrack256'))
# trackers.extend(trackerlist(name='timostrack', parameter_name='vitb_256_mae_ce_96x1_ep300-lasot-10', dataset_name=dataset_name,
#                             run_ids=None, display_name='TIMOSTrack256-10'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_96x1_ep300-lasotext', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_got10k_ep100', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384'))


dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec','prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),skip_missing_seq=True)
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
