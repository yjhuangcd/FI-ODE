from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Type, Any, Dict
from omegaconf import II, MISSING
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import hydra

ROOT = Path(__file__).parent
run_data_root = ROOT / 'run_data'


@dataclass
class DATASET:
    name: str
    IN_CHANNEL: int
    N_CLASSES: int
    IMG_SIZE: Tuple[int]
    data_root: str = II('data_root')


@dataclass
class FashionMNIST(DATASET):
    _target_: str = "dataset_loaders.load_fashion_mnist"
    name: str = "FashionMNIST"
    IN_CHANNEL: int = 1
    N_CLASSES: int = 10
    IMG_SIZE: Tuple[int] = field(default_factory=lambda: (28, 28))

@dataclass
class MNIST(DATASET):
    _target_: str = "dataset_loaders.load_mnist"
    name: str = "MNIST"
    IN_CHANNEL: int = 1
    N_CLASSES: int = 10
    IMG_SIZE: Tuple[int] = field(default_factory=lambda: (28, 28))
    MU: Tuple[float] = field(default_factory=lambda: [0.1307])
    STD: Tuple[float] = field(default_factory=lambda: [0.3081])

@dataclass
class CIFAR3(DATASET):
    _target_: str = "dataset_loaders.load_CIFAR3"
    name: str = "CIFAR3"
    IN_CHANNEL: int = 3
    N_CLASSES: int = 3
    IMG_SIZE: Tuple[int] = field(default_factory=lambda: (32, 32))
    MU: Tuple[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: Tuple[float] = field(default_factory=lambda: [0.225, 0.225, 0.225])

@dataclass
class CIFAR10(DATASET):
    _target_: str = "dataset_loaders.load_CIFAR10"
    name: str = "CIFAR10"
    IN_CHANNEL: int = 3
    N_CLASSES: int = 10
    IMG_SIZE: Tuple[int] = field(default_factory=lambda: (32, 32))
    MU: Tuple[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: Tuple[float] = field(default_factory=lambda: [0.225, 0.225, 0.225])

@dataclass
class Output:
    _target_: str  = "dynamics.output_coordinates.DefaultOutputFun"

@dataclass
class FirstNOutput:
    _target_: str = "dynamics.output_coordinates.FirstNOutput"
    out_size:int = II('dataset.N_CLASSES')

@dataclass
class DefaultInitFun:
    _target_: str = "dynamics.init_coordinates.DefaultInitFun"
    h_dims: Tuple[int] = field(default_factory=lambda: (II("dataset.N_CLASSES"),))
    param_map: Optional[Any] = MISSING

@dataclass
class UniformInitFun(DefaultInitFun):
    _target_: str = "dynamics.init_coordinates.UniformInitFun"

@dataclass
class OrthoClassDynProjectSimplex:
    _target_: str  = "dynamics.classification.OrthoClassDynProjectSimplex"
    n_hidden: int = II('..init_fun.h_dims[0]')
    activation: str = 'GroupSort'
    dropout:float = 0.5
    mlp_size:int = 128
    kappa:float = 5.
    kappa_length:int=0
    alpha_1:float = 5.
    alpha_2:float = 5.
    scale_nominal:bool = False

@dataclass
class OrthoClassDynProjectSimplexLips(OrthoClassDynProjectSimplex):
    _target_: str  = "dynamics.classification.OrthoClassDynProjectSimplexLips"
    sigma_1:float = 0.02
    x_dim: int=10
    cayley: bool=True

@dataclass
class ClassicalModel:
    # name:str = MISSING
    n_in_channels:int = II('dataset.IN_CHANNEL')
    n_outputs:int = II('dataset.N_CLASSES')


@dataclass
class CIFAR_4C3F:
    _target_: str = "models.make_4C3F"
    mu: Tuple[int] = II('dataset.MU')
    std: Tuple[int] = II('dataset.STD')
    out_dim: int = 128
    act: str = "GroupSort"


@dataclass
class CIFAR_4C3F_nolips:
    _target_: str = "models.make_4C3F_nolips"
    mu: Tuple[int] = II('dataset.MU')
    std: Tuple[int] = II('dataset.STD')
    out_dim: int = 128
    act: str = "ReLU"


@dataclass
class CIFAR_6C2F:
    _target_: str = "models.make_6C2F"
    mu: Tuple[int] = II('dataset.MU')
    std: Tuple[int] = II('dataset.STD')


@dataclass
class ORTHO_KWLarge_Concat(ClassicalModel):
    # name = "RESNET18"
    _target_: str = "models.make_ortho_KWLarge_Concat"
    mu: Tuple[int] = II('dataset.MU')
    std: Tuple[int] = II('dataset.STD')
    out_dim: int = 128
    act: str = "GroupSort"

@dataclass
class ORTHO_KWLargeMNIST_Concat(ClassicalModel):
    _target_: str = "models.make_ortho_KWLargeMNIST_Concat"
    mu: Tuple[int] = II('dataset.MU')
    std: Tuple[int] = II('dataset.STD')
    out_dim: int = 128
    act: str = "GroupSort"

@dataclass
class ORTHO_KWLarge_Concat_test(ClassicalModel):
    _target_: str = "models.make_ortho_KWLarge_Concat_test"
    mu: Tuple[int] = II('dataset.MU')
    std: Tuple[int] = II('dataset.STD')
    out_dim: int = 128
    act: str = "GroupSort"

@dataclass
class ORTHO_KWLargeMNIST_Concat_test(ClassicalModel):
    _target_: str = "models.make_ortho_KWLargeMNIST_Concat_test"
    mu: Tuple[int] = II('dataset.MU')
    std: Tuple[int] = II('dataset.STD')
    out_dim: int = 128
    act: str = "GroupSort"


@dataclass
class MSELoss:
    _target_: str = 'lya_cands.MSELoss'
    on_simplex: bool = II('..simplex')
    num_class: int = II('dataset.N_CLASSES')


@dataclass
class CompositeDynCrossEntropy:
    _target_: str = 'lya_cands.CompositeDynCrossEntropy'
    on_simplex: bool = II('..simplex')
    norm_type: str = 'L1'


@dataclass
class DynCrossEntropy:
    _target_: str = 'lya_cands.DynCrossEntropy'
    on_simplex: bool = II('..simplex')


@dataclass
class OnemEtay:
    _target_: str = 'lya_cands.OnemEtay'
    on_simplex: bool = II('..simplex')


@dataclass
class DecisionBoundary:
    _target_: str = 'lya_cands.DecisionBoundary'
    on_simplex: bool = II('..simplex')
    log_mode: bool = False
    num_class: int = II('dataset.N_CLASSES')

@dataclass
class GeneralModule:
    decay_epochs: List[int] = field(default_factory=lambda: [30, 60, 90])
    weight_decay: float = 0.0
    lr: float = 1e-3
    opt_name: str = 'SGD'
    momentum: float = 0.9
    beta1:float = 0.9
    beta2:float = 0.999
    scheduler_name: str = 'cos_anneal'
    max_epochs:int=200
    warmup:int=20
    adv_train:bool=False
    eps:float = 0.5
    norm: str= 'L2'
    act: str = 'relu'
    fix_backbone:bool=False
    val_adv:bool=True


@dataclass
class ClassicalModule(GeneralModule):
    _target_: str  = "pl_modules.ClassicalLearning"
    model:ClassicalModel = MISSING


@dataclass
class ODEModule(GeneralModule):
    _target_: str  = "pl_modules.ODELearning"
    dynamics:Any = MISSING
    output:Any = MISSING
    init_fun:Any = MISSING
    n_input: int = II('dataset.IN_CHANNEL')
    n_output: int = II('dataset.N_CLASSES')
    t_max:float = 1.0
    train_ode_solver: str = 'dopri5'
    train_ode_tol: float = 1e-7
    val_ode_solver: str = 'dopri5'
    val_ode_tol: float = 1e-7
    simplex:bool = False


@dataclass
class Lyapunov(ODEModule):
    _target_: str  = "pl_modules.LyapunovLearning"
    order: int = 1
    h_sample_size: int = 128
    h_dist_lim:float = 30
    sampler: Any = MISSING
    sampler_scheduler: Any = MISSING
    lya_cand: Any = MISSING
    barrier_loss:bool = False
    lips_train:bool = False
    train_ode:bool = False
    train_ode_epoch:int = 50
    relax_exp_stable:bool = False
    scaleLeps:float = 3.
    epoch_off_scale: int = 10
    lips_warmup: int = 0

@dataclass
class LinearScheduler:
    _target_: str = "sampling.sampler_schedulers.LinearScheduler"
    rate: float = 1.0
    bias: float = 0.0
    clamp: str = "min"
    clamp_val: float = 0.
    start: int = 0

@dataclass
class ConstantScheduler:
    _target_: str = "sampling.sampler_schedulers.ConstantScheduler"
    constant: float = 1.0


@dataclass
class SwitchScheduler:
    _target_: str = "sampling.sampler_schedulers.SwitchScheduler"
    start: float = 0.0
    end: float = 1.0
    trigger: float = 1.0

@dataclass
class CompositeSamplerScheduler:
    _target_: str = "sampling.sampler_schedulers.CompositeSamplerScheduler"
    schedulers: List[Any] = II('oc.dict.values:_sch_callback_dict')
    scheduler_weights: List[float] = MISSING

@dataclass
class AbstractSampler:
    h_dims: Tuple[int] = field(default_factory=lambda: (II("dataset.N_CLASSES"),))

@dataclass
class UniformSimplexSampling(AbstractSampler):
    _target_: str = "sampling.sampler.UniformSimplexSampling"

@dataclass
class BandSimplexSampling(AbstractSampler):
    _target_: str = "sampling.sampler.BandSimplexSampling"

@dataclass
class TrajectorySampler(AbstractSampler):
    _target_: str = "sampling.sampler.TrajectorySampler"

@dataclass
class DecisionBoundarySampling(AbstractSampler):
    _target_: str = "sampling.sampler.DecisionBoundarySampling"

@dataclass
class CorrectConeSampling(AbstractSampler):
    _target_: str = "sampling.sampler.CorrectConeSampling"

@dataclass
class ProjectedBiasedHyperSphereSampling(AbstractSampler):
    _target_: str = "sampling.sampler.ProjectedBiasedHyperSphereSampling"
    n_output: int = II('dataset.N_CLASSES')
    h_dist_lim: float = II('module.h_dist_lim')

@dataclass
class ProjectedHyperCubeSampling(AbstractSampler):
    _target_: str = "sampling.sampler.ProjectedHyperCubeSampling"
    h_dist_lim: float = II('module.h_dist_lim')

@dataclass
class CompositeSampler:
    _target_: str = "sampling.sampler.CompositeSampler"
    samplers: List = II('oc.dict.values:_sampler_callback_dict')
    h_dims: Tuple[int] = field(default_factory=lambda: (II("dataset.N_CLASSES"),))

@dataclass
class ExpCfg:
    # defaults:List[Any] = field(default_factory=lambda: ExpCfgDefaults)
    dataset: DATASET = MISSING
    savedir: str = run_data_root
    data_root: str = ROOT / 'data'
    batch_size: int = 32
    val_batch_size: int = 32
    data_loader_workers: int = 4
    prefetch_factor: int = 4
    disable_logs: bool = False
    module: GeneralModule = MISSING
    gpus:int = 0
    seed: int = 0

@dataclass
class RobustExpCfg(ExpCfg):
    model_file: str = MISSING
    norm:str = "2" # only 2 or inf
    eps:float = 0.5
    download:bool = True

@dataclass
class CertifyExpCfg(RobustExpCfg):
    model_file: str = MISSING
    norm:str = "2" # only 2 or inf
    eps:float = 0.141
    download:bool = True
    kappa: float = 0.2
    T: int = 40
    batches: int = 10
    load_grid: bool = False
    grid_name: str = "grid.pt"
    m: int = 100
    start_ind: int=0
    end_ind: int=10000

cs = ConfigStore.instance()


# cs.store(group='dataset', name='ImageNet', node=ImageNet)
cs.store(group='dataset', name='MNIST', node=MNIST)
cs.store(group='dataset', name='FashionMNIST', node=FashionMNIST)
cs.store(group='dataset', name='CIFAR3', node=CIFAR3)
cs.store(group='dataset', name='CIFAR10', node=CIFAR10)

cs.store(group='module/init_fun/param_map', name="CIFAR_4C3F", node=CIFAR_4C3F)
cs.store(group='module/init_fun/param_map', name="CIFAR_4C3F_nolips", node=CIFAR_4C3F_nolips)
cs.store(group='module/init_fun/param_map', name="CIFAR_6C2F", node=CIFAR_6C2F)
cs.store(group='module/init_fun/param_map', name="ORTHO_KWLarge_Concat", node=ORTHO_KWLarge_Concat)
cs.store(group='module/init_fun/param_map', name="ORTHO_KWLargeMNIST_Concat", node=ORTHO_KWLargeMNIST_Concat)
cs.store(group='module/init_fun/param_map', name="ORTHO_KWLargeMNIST_Concat_test", node=ORTHO_KWLargeMNIST_Concat_test)
cs.store(group='module/init_fun/param_map', name="ORTHO_KWLarge_Concat_test", node=ORTHO_KWLarge_Concat_test)
cs.store(group='module/lya_cand', name="MSELoss", node=MSELoss)
cs.store(group='module/lya_cand', name="CompositeDynCrossEntropy", node=CompositeDynCrossEntropy)
cs.store(group='module/lya_cand', name="DynCrossEntropy", node=DynCrossEntropy)
cs.store(group='module/lya_cand', name="OnemEtay", node=OnemEtay)
cs.store(group='module/lya_cand', name="DecisionBoundary", node=DecisionBoundary)
cs.store(group='module/init_fun', name="DefaultInitFun", node=DefaultInitFun)
cs.store(group='module/init_fun', name="UniformInitFun", node=UniformInitFun)
cs.store(group='module/dynamics', name="OrthoClassDynProjectSimplex", node=OrthoClassDynProjectSimplex)
cs.store(group='module/dynamics', name="OrthoClassDynProjectSimplexLips", node=OrthoClassDynProjectSimplexLips)
cs.store(group='module/sampler', name="CompositeSampler", node=CompositeSampler)
cs.store(group='module/sampler', name="UniformSimplexSampling", node=UniformSimplexSampling)
cs.store(group='module/sampler', name="BandSimplexSampling", node=BandSimplexSampling)
cs.store(group='module/sampler', name="ProjectedBiasedHyperSphereSampling", node=ProjectedBiasedHyperSphereSampling)
cs.store(group='module/sampler', name="ProjectedHyperCubeSampling", node=ProjectedHyperCubeSampling)
cs.store(group='module/sampler', name="TrajectorySampler", node=TrajectorySampler)
cs.store(group='module/sampler', name="CorrectConeSampling", node=CorrectConeSampling)
cs.store(group='module/sampler', name="DecisionBoundarySampling", node=DecisionBoundarySampling)
cs.store(group='module/sampler_scheduler', name="CompositeSamplerScheduler", node=CompositeSamplerScheduler)
cs.store(group='module/sampler_scheduler', name="LinearScheduler", node=LinearScheduler)
cs.store(group='module/sampler_scheduler', name="ConstantScheduler", node=ConstantScheduler)
cs.store(group='module/sampler_scheduler', name="SwitchScheduler", node=SwitchScheduler)
cs.store(group='module/output', name="Output", node=Output)
cs.store(group='module/output', name="FirstNOutput", node=FirstNOutput)
cs.store(group='module', name="ClassicalModule", node=ClassicalModule)
cs.store(group='module', name="ODEModule", node=ODEModule)
cs.store(group='module', name="Lyapunov", node=Lyapunov)
cs.store(name='default', node=ExpCfg)
cs.store(name='robust', node=RobustExpCfg)
cs.store(name='certify', node=CertifyExpCfg)