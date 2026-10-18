import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import shutil
from torch import nn
from tqdm import tqdm
from torch.distributions import Normal
from gymnasium.wrappers import RecordVideo
from pathlib import Path
from torch.optim.lr_scheduler import LinearLR
class Actor(nn.Module):
    def __init__(self,din,dout):
        super().__init__()
        self.backbone=nn.Sequential(
            nn.Linear(din,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU()
        )
        self.mean=nn.Linear(512,dout)
        self.logstd=nn.Parameter(torch.full((dout,),-0.5))
    def forward(self,x):
        features=self.backbone(x)
        mu=F.tanh(self.mean(features))
        logstd=self.logstd.expand_as(mu)
        return mu, logstd

class Critic(nn.Module):
    def __init__(self,din,dout=1):
        super().__init__()
        self.backbone=nn.Sequential(
            nn.Linear(din,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,dout)
        )
    def forward(self,x):
        return self.backbone(x)

def test(model,tenv,device="cuda"):

  don=False
  ts,_=tenv.reset()
  gt=0
  with torch.no_grad():
    while not don:
      mean,_=model(torch.tensor(ts,dtype=torch.float32,device=device))
      ts,tr,don,t,_=tenv.step(mean.cpu().numpy())
      gt+=tr
      if t:
        break
  tenv.close()

  return gt
if __name__ == "__main__":
  dob=24
  daction=4
  actor=Actor(dob,daction)
  critic=Critic(dob)
  device="cuda" if torch.cuda.is_available() else "cpu"
  actor.to(device)
  critic.to(device)
  alr=3e-4
  clr=1e-3
  atlr=0.3
  ctlr=0.3
  actor_optimizer=torch.optim.AdamW(actor.parameters(),lr=alr)
  critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=clr)



  env_name="BipedalWalker-v3"
  num_envs=5
  envs=gym.make_vec(env_name,num_envs,vectorization_mode="async")
  tenv=gym.envs.make(env_name,render_mode="rgb_array")
  tenv=RecordVideo(tenv,video_folder="./videos",episode_trigger=lambda x:True)
  total_steps=5000000
  steps=4000
  gamma=0.99
  lam=0.9
  epochs=10
  eps=0.2
  w=0.01
  num_rollouts=total_steps//num_envs//steps
  batch_size=256
  s,_=envs.reset()
  als=LinearLR(actor_optimizer,1,atlr,total_iters=num_rollouts*epochs)
  cls=LinearLR(critic_optimizer,1,ctlr,total_iters=num_rollouts*epochs)

  # rollouts
  for ro in tqdm(range(num_rollouts)):
    states=np.zeros((num_envs,steps,dob))
    actions=np.zeros((num_envs,steps,daction))
    rewards=np.zeros((num_envs,steps))
    dones=np.zeros((num_envs,steps))
    logprobs=np.zeros((num_envs,steps))
    values=np.zeros((num_envs,steps))
    with torch.no_grad():
        for i in range(steps):
            states[:,i]=s
            stensor=torch.tensor(s,dtype=torch.float32,device=device)
            vs=critic(stensor).reshape(-1).cpu().numpy()
            values[:,i]=vs
            mu, logstd=actor(stensor)
            dist=Normal(mu,logstd.exp())
            action=dist.sample()
            actions[:,i]=action.cpu().numpy()
            logprobs[:,i]=dist.log_prob(action).cpu().sum(-1).numpy()
            s,r,done,truncated,info=envs.step(action.clip(-1,1).cpu().numpy())
            dones[:,i]=done|truncated
            rewards[:,i]=r

        # bootstrap at termination or truncation
            if truncated.sum():
                val=critic(torch.tensor(s[truncated],dtype=torch.float32,device=device)).reshape(-1).cpu().numpy()
                rewards[:,i][truncated]+=gamma*val
        next_val=critic(torch.tensor(s,dtype=torch.float32,device=device))
        next_val[done]=0
        next_val[truncated]=0
        next_val=next_val.reshape(-1).cpu().numpy()
        gaes=np.zeros((num_envs,steps))
        dones=dones.astype(np.bool)

        # compute GAE
        for j in reversed(range(steps)):
          mask=dones[:,j]
          if j==steps-1:
            gaes[:,j][mask]=rewards[:,j][mask]-values[:,j][mask]
            gaes[:,j][~mask]=rewards[:,j][~mask]+gamma*next_val[~mask]-values[:,j][~mask]
          else:
            gaes[:,j][mask]=rewards[:,j][mask]-values[:,j][mask]
            gaes[:,j][~mask]=rewards[:,j][~mask]+gamma*values[:,j+1][~mask]-values[:,j][~mask]+gamma*lam*gaes[:,j+1][~mask]
        returns=torch.tensor(gaes+values)
        gaes=((gaes-gaes.mean())/(gaes.std()+1e-8))

        state_tensor=torch.tensor(states,dtype=torch.float32)
        action_tensor=torch.tensor(actions,dtype=torch.float32)
        value_tensor=torch.tensor(values,dtype=torch.float32)
        logprob_tensor=torch.tensor(logprobs,dtype=torch.float32)
        gae_tensor=torch.tensor(gaes).to(device)

    # training
    for e in range(epochs):
      indices=np.arange(steps)
      np.random.shuffle(indices)
      for b in range(steps//batch_size):
        start=b*batch_size
        end=start+batch_size
        batch=indices[start:end]
        batch_state=state_tensor[:,batch].to(device)
        batch_returns=returns[:,batch].to(device)
        batch_actions=action_tensor[:,batch].to(device)
        batch_logprobs=logprob_tensor[:,batch].to(device)
        batch_gae=gae_tensor[:,batch].to(device)
        mu,logstd=actor(batch_state)
        pi=Normal(mu,logstd.exp())
        new_logprob=pi.log_prob(batch_actions).sum(-1)
        ratio=torch.exp(new_logprob-batch_logprobs)
        actor_loss=-torch.min(ratio*batch_gae,torch.clip(ratio,1-eps,1+eps)*batch_gae).mean()
        new_values=critic(batch_state).squeeze(-1)
        critic_loss=((new_values-batch_returns)**2).mean()
        entropy=pi.entropy().sum(-1).mean()
        loss=actor_loss+critic_loss-w*entropy
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
      als.step()
      cls.step()
    if ro %5==0 and ro!=0:
      test_reward=test(actor,tenv,device)
      print(f"current actor learning rate:{als.get_last_lr()}; current critic learning rate: {cls.get_last_lr()}; test reward: {test_reward}")
