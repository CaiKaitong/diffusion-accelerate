#!/usr/bin/env python3
import socket
import torch
import numpy as np
import cv2
import base64
import threading
import time
from pathlib import Path
from typing import Optional, Dict
import hydra
import dill
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict

from server_config import (
    SERVER_IP, SERVER_PORT, CHECKPOINT_PATH, USE_EMA,
    DEVICE, SCHEDULER_TYPE, NUM_INFERENCE_STEPS, INFERENCE_FREQ,
    SOCKET_TIMEOUT, BUFFER_SIZE, ENCODING, MAX_CLIENTS, VERBOSE
)


class DPInferenceServerSSH:
    """Diffusion Policy Inference Server (SSH Tunnel Version)"""
    
    def __init__(self,
                 checkpoint_path: str = CHECKPOINT_PATH,
                 use_ema: bool = USE_EMA,
                 device: str = DEVICE,
                 scheduler_type: str = SCHEDULER_TYPE,
                 num_inference_steps: int = NUM_INFERENCE_STEPS,
                 inference_freq: float = INFERENCE_FREQ,
                 server_ip: str = SERVER_IP,
                 server_port: int = SERVER_PORT,
                 max_clients: int = MAX_CLIENTS,
                 verbose: bool = VERBOSE):
        
        self.checkpoint_path = checkpoint_path
        self.use_ema = use_ema
        self.device = device
        self.scheduler_type = scheduler_type.upper()
        self.num_inference_steps = num_inference_steps
        self.inference_freq = inference_freq
        self.server_ip = server_ip
        self.server_port = server_port
        self.max_clients = max_clients
        self.verbose = verbose
        
        self.policy = None
        self.cfg = None
        self.running = False
        self.expected_image_shape = None
        self.obs_keys = None
        
        self.obs_history = {
            'images': [],
            'states': []
        }
        
        self.inference_log = {
            'steps': []
        }
        self._load_model()
    
    def _load_model(self):
        payload = torch.load(open(self.checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        self.policy = workspace.model
        if self.use_ema:
            self.policy = workspace.ema_model
        
        self.policy.eval().to(self.device)
        
        if self.scheduler_type == "DDIM":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            self.policy.noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
        elif self.scheduler_type == "DDPM":
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            self.policy.noise_scheduler = DDPMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon',
                variance_type='fixed_small'
            )
        else:
            raise ValueError(f"Unsupported Scheduler type: {self.scheduler_type}, use 'DDIM' or 'DDPM'")
        
        self.policy.num_inference_steps = self.num_inference_steps
        
        shape_meta = self.cfg.task.shape_meta
        rgb_keys = [k for k, v in shape_meta['obs'].items() if v.get('type') == 'rgb']
        lowdim_keys = [k for k, v in shape_meta['obs'].items() if v.get('type') == 'low_dim']

        self.obs_keys = {
            'rgb': rgb_keys[0] if rgb_keys else None,
            'lowdim': lowdim_keys
        }

        if self.obs_keys['rgb']:
            image_shape = shape_meta['obs'][self.obs_keys['rgb']]['shape']
            self.expected_image_shape = (image_shape[1], image_shape[2])
        
        self._warmup_model()
    
    def _warmup_model(self):
        batch_size = 1
        n_obs_steps = self.policy.n_obs_steps
        
        shape_meta = self.cfg.task.shape_meta
        obs_keys = shape_meta['obs']
        
        rgb_key = None
        for key, spec in obs_keys.items():
            if spec.get('type') == 'rgb':
                rgb_key = key
                break
        
        if rgb_key is None:
            raise ValueError("RGB observation key not found")
        
        image_shape = shape_meta['obs'][rgb_key]['shape']
        
        dummy_image = torch.randn(
            batch_size, n_obs_steps, *image_shape,
            device=self.device, dtype=torch.float32
        )
        
        obs_dict = {rgb_key: dummy_image}
        
        for key, spec in obs_keys.items():
            if spec.get('type') == 'low_dim':
                low_dim_shape = spec['shape']
                dummy_low_dim = torch.randn(
                    batch_size, n_obs_steps, *low_dim_shape,
                    device=self.device, dtype=torch.float32
                )
                obs_dict[key] = dummy_low_dim
        
        with torch.no_grad():
            self.policy.reset()
            result = self.policy.predict_action(obs_dict)
            action = result['action'][0].detach().to('cpu').numpy()
    
    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.server_ip, self.server_port))
        server_socket.listen(self.max_clients)
        self.running = True
        
        while self.running:
            try:
                client_socket, client_addr = server_socket.accept()
                self._handle_client(client_socket, client_addr)
            except KeyboardInterrupt:
                break
        
        server_socket.close()
        self._save_inference_log()
    
    def _handle_client(self, client_socket: socket.socket, client_addr: tuple):
        client_socket.settimeout(SOCKET_TIMEOUT)
        buffer = b''
        
        while self.running:
            try:
                data = client_socket.recv(BUFFER_SIZE)
                if not data:
                    break
                
                buffer += data
                
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    msg = line.decode(ENCODING).strip()
                    
                    if msg:
                        self._process_message(client_socket, msg)
            
            except socket.timeout:
                continue
            except:
                break
        
        client_socket.close()
    
    def _process_message(self, client_socket: socket.socket, message: str):
        data = json.loads(message)
        
        if data.get('type') == 'reset':
            self.policy.reset()
            response = {'type': 'reset_ack'}
            msg = json.dumps(response) + '\n'
            client_socket.sendall(msg.encode(ENCODING))
            
        elif data.get('type') == 'observation':
            images_b64 = data.get('images', [])
            poses_list = data.get('poses', [])
            grippers_list = data.get('grippers', [])
            timestamps = np.array(data.get('timestamps', []), dtype=np.float32)

            images = []
            for img_b64 in images_b64:
                img_data = base64.b64decode(img_b64)
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.expected_image_shape is not None:
                    expected_h, expected_w = self.expected_image_shape
                    current_h, current_w = image.shape[:2]

                    if (current_h, current_w) != (expected_h, expected_w):
                        image = cv2.resize(image, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)

                images.append(image)

            poses = np.array(poses_list, dtype=np.float32)
            grippers = np.array(grippers_list, dtype=np.float32)

            env_obs = {}

            if self.obs_keys['rgb']:
                env_obs[self.obs_keys['rgb']] = np.stack(images, axis=0).astype(np.uint8)

            shape_meta = self.cfg.task.shape_meta
            for lowdim_key in self.obs_keys['lowdim']:
                if 'pose' in lowdim_key.lower():
                    env_obs[lowdim_key] = poses
                elif 'gripper' in lowdim_key.lower():
                    env_obs[lowdim_key] = grippers

            action = self._infer_action(env_obs, timestamps)
            
            response = {
                'type': 'action',
                'action': action.tolist()
            }
            
            msg = json.dumps(response) + '\n'
            client_socket.sendall(msg.encode(ENCODING))
    
    def reset_obs_history(self):
        self.obs_history['images'].clear()
        self.obs_history['states'].clear()
    
    def _infer_action(self, env_obs: dict, timestamps: np.ndarray) -> np.ndarray:
        lowdim_obs_list = []
        for lowdim_key in self.obs_keys['lowdim']:
            if lowdim_key in env_obs:
                lowdim_obs_list.append(env_obs[lowdim_key][-1])

        last_state = np.concatenate(lowdim_obs_list) if lowdim_obs_list else np.array([])
        
        current_step = {
            'step': len(self.inference_log['steps']),
            'input': {
                'state': last_state.astype(np.float32).tolist(),
                'n_obs_steps': len(timestamps),
                'timestamp': float(timestamps[-1]) if len(timestamps) > 0 else 0.0
            }
        }
        
        shape_meta = self.cfg.task.shape_meta
        env_obs['timestamp'] = timestamps

        obs_dict_np = get_real_obs_dict(
            env_obs=env_obs,
            shape_meta=shape_meta
        )
        
        obs_dict = dict_apply(obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            action = result['action'][0].detach().to('cpu').numpy()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time_ms = (time.time() - start_time) * 1000
        action_with_gripper = action
        
        current_step['action'] = {
            'values': action_with_gripper.astype(np.float32).tolist(),
            'shape': list(action_with_gripper.shape)
        }
        
        self.inference_log['steps'].append(current_step)
        return action_with_gripper.astype(np.float32)
    
    def _save_inference_log(self):
        import json
        from datetime import datetime
        
        log_dir = Path(__file__).parent / "log"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"inference_log_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.inference_log, f, indent=2)


if __name__ == "__main__":
    server = DPInferenceServerSSH()
    server.start()
