# Generate a graphical trace of multiple runs.

import json
import os
import zlib

import boto3
import wandb

from metta.agent.policy_store import PolicyRecord
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulator import Simulator
from metta.util.wandb.wandb_context import WandbContext


class ReplayHelper:
    """Helper class for generating and uploading replays."""

    def __init__(self, config: SimulationConfig, policy_record: PolicyRecord, wandb_run: WandbContext):
        self.config = config
        self.policy_record = policy_record
        self.wandb_run = wandb_run

        self.s3_client = boto3.client("s3")

    def _add_sequence_key(self, grid_object: dict, key: str, step: int, value):
        """Add a key to the replay that is a sequence of values."""
        if key not in grid_object:
            # Add new key.
            grid_object[key] = [[step, value]]
        else:
            # Only add new entry if it has changed:
            if grid_object[key][-1][1] != value:
                grid_object[key].append([step, value])

    def generate_replay(self, replay_path: str):
        """Generate a replay and save it to a file."""
        simulator = Simulator(self.config, self.policy_record)

        # Debug info about action space
        print(f"Action space: {simulator.env.action_space}")
        print(f"Action names: {simulator.env.action_names()}")

        grid_objects = []

        replay = {
            "version": 1,
            "action_names": simulator.env.action_names(),
            "object_types": [],
            "map_size": [simulator.env.map_width, simulator.env.map_height],
            "num_agents": simulator.num_agents,
            "max_steps": simulator.num_steps,
            "grid_objects": grid_objects,
        }

        replay["object_types"] = simulator.env.object_type_names()

        step = 0
        while not simulator.done():
            actions = simulator.actions()

            # Add debug info about actions
            print(f"Step {step} - Action shape: {actions.shape}")
            print(f"Step {step} - Action values: {actions}")
            print(f"Step {step} - Action type: {actions.dtype}")

            # Check if actions are within bounds before proceeding
            if hasattr(simulator.env, "action_space"):
                try:
                    # Attempt to validate actions against action space
                    actions_array = actions.cpu().numpy()
                    print(f"Step {step} - Action array shape: {actions_array.shape}")
                    print(f"Step {step} - Action array values: {actions_array}")

                    # If discrete action space, check if all values are within bounds
                    if hasattr(simulator.env.action_space, "n"):
                        action_max = simulator.env.action_space.n - 1
                        action_min = 0
                        max_val = actions_array.max()
                        min_val = actions_array.min()
                        print(f"Step {step} - Action space bounds: min={action_min}, max={action_max}")
                        print(f"Step {step} - Actual action bounds: min={min_val}, max={max_val}")

                        if min_val < action_min or max_val > action_max:
                            print(f"WARNING: Actions out of bounds at step {step}!")
                except Exception as e:
                    print(f"Error validating actions at step {step}: {e}")

            for i, grid_object in enumerate(simulator.grid_objects()):
                if len(grid_objects) <= i:
                    # Add new grid object.
                    grid_objects.append({})
                for key, value in grid_object.items():
                    self._add_sequence_key(grid_objects[i], key, step, value)

                if "agent_id" in grid_object:
                    agent_id = grid_object["agent_id"]
                    # Print agent action info
                    print(f"Step {step} - Agent {agent_id} action: {actions_array[agent_id].tolist()}")

                    self._add_sequence_key(grid_objects[i], "action", step, actions_array[agent_id].tolist())
                    self._add_sequence_key(
                        grid_objects[i], "action_success", step, bool(simulator.env.action_success[agent_id])
                    )
                    self._add_sequence_key(grid_objects[i], "reward", step, simulator.rewards[agent_id].item())
                    self._add_sequence_key(
                        grid_objects[i], "total_reward", step, simulator.total_rewards[agent_id].item()
                    )

            try:
                simulator.step(actions)
            except Exception as e:
                print(f"Error in simulator.step at step {step}: {e}")
                # Print additional debugging information
                print(f"Full action tensor: {actions}")
                raise

            step += 1

        replay["max_steps"] = step

        # Trim value changes to make them more compact.
        for grid_object in grid_objects:
            for key, changes in grid_object.items():
                if len(changes) == 1:
                    grid_object[key] = changes[0][1]

        # Compress it with deflate.
        replay_data = json.dumps(replay)  # Convert to JSON string
        replay_bytes = replay_data.encode("utf-8")  # Encode to bytes
        compressed_data = zlib.compress(replay_bytes)  # Compress the bytes

        # Make sure the directory exists.
        os.makedirs(os.path.dirname(replay_path), exist_ok=True)

        with open(replay_path, "wb") as f:
            f.write(compressed_data)

    def upload_replay(self, replay_path: str, replay_url: str, epoch: int):
        """Upload the replay to S3 and log the link to WandB."""
        s3_bucket = "softmax-public"
        self.s3_client.upload_file(
            Filename=replay_path, Bucket=s3_bucket, Key=replay_url, ExtraArgs={"ContentType": "application/x-compress"}
        )
        link = f"https://{s3_bucket}.s3.us-east-1.amazonaws.com/{replay_url}"

        # Log the link to WandB
        player_url = "https://metta-ai.github.io/metta/?replayUrl=" + link
        link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
        self.wandb_run.log(link_summary)

    def generate_and_upload_replay(self, epoch: int, run_dir: str, run: str, dry_run: bool = False):
        """Generate a replay and upload it to S3 and log the link to WandB."""
        replay_path = f"{run_dir}/replays/replay.{epoch}.json.z"
        self.generate_replay(replay_path)

        replay_url = f"replays/{run}/replay.{epoch}.json.z"
        if not dry_run:
            self.upload_replay(replay_path, replay_url, epoch)
