# Import datasets
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from datasets.interface import TrajectoryDataset
from datasets.nuScenes.nuScenes_raster import NuScenesRaster
from datasets.nuScenes.nuScenes_vector import NuScenesVector
from datasets.nuScenes.nuScenes_graphs import NuScenesGraphs

# Import model
from models.model import PredictionModel
from models.encoders.ac_encoder import ACEncoder
from models.aggregators.ac_aggregator import ACInteraction
from models.decoders.query_tr import QueryTr


# Import metrics
from metrics.min_ade import MinADEK
from metrics.min_fde import MinFDEK
from metrics.miss_rate import MissRateK
from metrics.LaplaceNLLLoss import LaplaceNLLLoss

from typing import List, Dict, Union


# Datasets
def initialize_dataset(dataset_type: str, args: List) -> TrajectoryDataset:
    """
    Helper function to initialize appropriate dataset by dataset type string
    """
    # TODO: Add more datasets as implemented
    dataset_classes = {'nuScenes_single_agent_raster': NuScenesRaster,
                       'nuScenes_single_agent_vector': NuScenesVector,
                       'nuScenes_single_agent_graphs': NuScenesGraphs,
                       }
    return dataset_classes[dataset_type](*args)


def get_specific_args(dataset_name: str, data_root: str, version: str = None) -> List:
    """
    Helper function to get dataset specific arguments.
    """
    # TODO: Add more datasets as implemented
    specific_args = []
    if dataset_name == 'nuScenes':
        ns = NuScenes(version, dataroot=data_root)
        pred_helper = PredictHelper(ns)
        specific_args.append(pred_helper)

    return specific_args


# Models
def initialize_prediction_model(encoder_type: str, aggregator_type: str, decoder_type: str,
                                encoder_args: Dict, aggregator_args: Union[Dict, None], decoder_args: Dict):
    """
    Helper function to initialize appropriate encoder, aggegator and decoder model
    """
    encoder = initialize_encoder(encoder_type, encoder_args)
    aggregator = initialize_aggregator(aggregator_type, aggregator_args)
    decoder = initialize_decoder(decoder_type, decoder_args)
    model = PredictionModel(encoder, aggregator, decoder)

    return model


def initialize_encoder(encoder_type: str, encoder_args: Dict):
    """
    Initialize appropriate encoder by type.
    """
    # TODO: Update as we add more encoder types
    encoder_mapping = {
        'ac_encoder': ACEncoder,
    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict, None]):
    """
    Initialize appropriate aggregator by type.
    """
    # TODO: Update as we add more aggregator types
    aggregator_mapping = {
        'ac_aggregator': ACInteraction,
    }

    if aggregator_args:
        return aggregator_mapping[aggregator_type](aggregator_args)
    else:
        return aggregator_mapping[aggregator_type]()


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # TODO: Update as we add more decoder types
    decoder_mapping = {
        'query': QueryTr,
    }

    return decoder_mapping[decoder_type](decoder_args)


# Metrics
def initialize_metric(metric_type: str, metric_args: Dict = None):
    """
    Initialize appropriate metric by type.
    """
    # TODO: Update as we add more metrics
    metric_mapping = {
        'min_ade_k': MinADEK,
        'min_fde_k': MinFDEK,
        'miss_rate_k': MissRateK,
        'LaplaceLoss': LaplaceNLLLoss,
    }

    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()
