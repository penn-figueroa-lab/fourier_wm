"""Contract/numerical checks for the unchanged algorithms and SWM glue."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from fourier.predictor import FourierDynamics
from fourier.training import endpoint_loss, world_model_loss, SIGReg
from fourier.world_model import FourierWorldModel

ROOT = Path(__file__).resolve().parents[1]

class Encoder(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.linear = nn.Linear(3,d)
    def forward(self, pixels, **kwargs):
        return SimpleNamespace(last_hidden_state=self.linear(
            pixels.mean((-1,-2))).unsqueeze(1))

def dynamics():
    torch.manual_seed(71)
    return FourierDynamics(latent_dim=4, action_dim=2, num_frequencies=5, init_scale=.03)

def test_original_definitions_unchanged():
    hashes=json.loads((ROOT/"tests/original_hashes.json").read_text())
    for original,migrated in [("model.py","predictor.py"),("training.py","training.py"),("sqp.py","sqp.py")]:
        expected=hashes[original]
        data=(ROOT/"src/fourier"/migrated).read_bytes()[:expected["bytes"]]
        assert hashlib.sha256(data).hexdigest()==expected["sha256"]

def test_endpoint_and_gradient_contract():
    model=dynamics()
    emb=torch.randn(2,4,4,requires_grad=True)
    controls=torch.randn(2,3,2,2)
    expected=[]
    for i in range(3):
        z=emb[:,i]
        for j in range(2): z=model(z,controls[:,i,j])
        expected.append(z)
    expected=(torch.stack(expected,1)-emb[:,1:]).square().mean()
    actual=endpoint_loss(model,emb,controls)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert emb.grad[:,0].abs().sum()>0
    assert emb.grad[:,-1].abs().sum()>0
    assert model.control_raw.grad.abs().sum()>0
    loss,logs=world_model_loss(model,emb,controls,SIGReg(num_proj=8))
    assert loss.requires_grad and all(not x.requires_grad for x in logs.values())

def test_exact_torch_jacobian():
    model=dynamics().double()
    z=torch.randn(4,dtype=torch.float64)
    u=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
    end,jac=model.terminal_with_action_jacobian(z,u)
    automatic=torch.autograd.functional.jacobian(
        lambda a:model.rollout(z,a)[-1],u).reshape(4,-1)
    torch.testing.assert_close(end,model.rollout(z,u)[-1])
    torch.testing.assert_close(jac,automatic,rtol=1e-10,atol=1e-10)

def test_inherited_rollout_and_action_gradient():
    model=FourierWorldModel(Encoder(),dynamics(),action_block=2)
    pixels=torch.randn(2,3,2,3,4,4)
    actions=torch.randn(2,3,3,4,requires_grad=True)
    history=torch.randn(2,3,1,4)
    output=model.rollout({"pixels":pixels,"action_history":history},actions)
    pred=output["predicted_emb"]
    assert pred.shape==(2,3,5,4)
    z=model.encode({"pixels":pixels[:,0]})["emb"][:,-1]
    z=z[:,None].expand(2,3,4)
    direct=model.predictor.rollout(z,actions.reshape(2,3,6,2))
    torch.testing.assert_close(pred[:,:,2:],direct[:,:,1::2],atol=1e-6,rtol=1e-5)
    pred[:,:,-1].square().sum().backward()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum()>0

def test_jax_cost_gradient_and_solve():
    import jax
    import jax.numpy as jnp
    from fourier.sqp import make_sqp,stack_parameters
    model=dynamics()
    z=torch.randn(2,4)
    u=torch.randn(2,3,2)*.2
    goal=torch.randn(2,4)
    p=stack_parameters(model.numpy_parameters())
    solve,cost=make_sqp(horizon=3,action_dim=2,iterations=3,smoothness=.1)
    def tcost(a):
        return (model.rollout(z,a)[:,-1]-goal).square().mean(-1)+.1*a.diff(dim=1).square().mean((1,2))
    np.testing.assert_allclose(cost(p,jnp.asarray(z),jnp.asarray(goal),jnp.asarray(u)),tcost(u).detach(),rtol=2e-5,atol=2e-6)
    u.requires_grad_(True)
    tg=torch.autograd.grad(tcost(u).sum(),u)[0]
    jg=jax.grad(lambda a:cost(p,jnp.asarray(z),jnp.asarray(goal),a).sum())(jnp.asarray(u.detach()))
    np.testing.assert_allclose(jg,tg,rtol=2e-4,atol=2e-6)
    initial=jnp.asarray(u.detach())
    final=solve(p,jnp.asarray(z),jnp.asarray(goal),initial)
    assert np.isfinite(final).all() and np.abs(final).max()<=3
    assert np.all(np.asarray(cost(p,jnp.asarray(z),jnp.asarray(goal),final))<=np.asarray(cost(p,jnp.asarray(z),jnp.asarray(goal),initial))+1e-6)

def test_solver_protocol_and_coordinate_conversion():
    from gymnasium.spaces import Box
    from stable_worldmodel.planning import ShootingCostEvaluator,GoalMSE
    from stable_worldmodel.policy import PlanConfig
    from stable_worldmodel.planning.solver.solver import Solver
    from fourier.sqp import SWMSQPSolver
    model=FourierWorldModel(Encoder(),dynamics(),action_block=2)
    cost=ShootingCostEvaluator(model,GoalMSE(reduction="mean"))
    solver=SWMSQPSolver(cost,iterations=0,device="cpu")
    solver.configure(action_space=Box(-1,1,shape=(2,2)),n_envs=2,
                     config=PlanConfig(horizon=2,receding_horizon=1,action_block=2))
    assert isinstance(solver,Solver)
    solver._eval_mean=np.array([.1,.2])
    solver._eval_scale=np.array([.5,2.])
    info={"pixels":torch.randn(2,1,3,4,4),
          "goal":torch.randn(2,1,3,4,4),"action":torch.zeros(2,1,2)}
    prefix=torch.tensor([[[.1,.2,.3,.4]],[[.5,.6,.7,.8]]])
    result=solver(info,init_action=prefix)
    expected=torch.cat([prefix.reshape(2,2,2),prefix.reshape(2,2,2)[:,-1:].expand(2,2,2)],1).reshape(2,2,4)
    torch.testing.assert_close(result["actions"],expected)

def test_config_targets():
    from hydra import initialize_config_dir,compose
    with initialize_config_dir(config_dir=str(ROOT/"configs"),version_base=None):
        cfg=compose(config_name="training")
        assert cfg.model._target_=="fourier.world_model.FourierWorldModel"
        assert cfg.model.predictor._target_=="fourier.predictor.FourierDynamics"
        assert "action_encoder" not in cfg.model
        assert "action" in cfg.data.dataset.keys_to_load
        planning=compose(config_name="planning")
        assert planning.solver._target_=="fourier.sqp.SWMSQPSolver"
        assert planning.plan_config.horizon*planning.plan_config.action_block==25
