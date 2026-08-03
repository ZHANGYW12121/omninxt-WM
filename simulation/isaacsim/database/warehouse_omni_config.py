#!/usr/bin/env python3
"""Overrides for the pedestrian-free Isaac Simple Warehouse Omni-Depth scene."""

from crowd_templates import CrowdSceneConfig


WAREHOUSE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/warehouse.usd"
)
WAREHOUSE_CENTER_SPAWN = [0.0, 0.0, 0.45]


def apply(config):
    """Mutate the already-loaded base config before pegasus_app imports it."""
    config.USD_PATH = WAREHOUSE_USD
    config.SPAWN_POS = list(WAREHOUSE_CENTER_SPAWN)

    # A broad navigation region around the official warehouse origin. It is
    # retained for manual/classic flight compatibility but no people are made.
    config.PEDESTRIAN_WALK_POLYGON = [
        (-12.0, -12.0),
        (12.0, -12.0),
        (12.0, 12.0),
        (-12.0, 12.0),
    ]
    config.DATASET_GOAL_X_RANGE = (-2.0, 2.0)
    config.DATASET_GOAL_Y_MIN = 10.0
    config.TARGET_POINT = [0.0, 10.0, config.CLASSIC_CRUISE_HEIGHT]
    config.SCENE_DRONE_X_RANGE = (0.0, 0.0)
    config.SCENE_DRONE_Y_RANGE = (0.0, 0.0)
    config.CROWD_RANDOMIZE_TEMPLATE = False
    config.CROWD_POOL_PEOPLE_COUNT = 0
    config.CROWD_TEMPLATE = {
        "num_people": 0,
        "valid_people_counts": (0,),
        "group_spacing": "close",
        "direction": "same_direction",
        "drone_distance": "near",
        "speed": "slow",
        "seed": 1,
    }
    config.ACTIVE_CROWD_TEMPLATE = dict(config.CROWD_TEMPLATE)
    config.ACTIVE_CROWD_SCENE = CrowdSceneConfig(
        key="warehouse_omni_no_people",
        num_people=0,
        group_spacing="close",
        direction="same_direction",
        drone_distance="near",
        speed="slow",
        seed=1,
        drone_spawn=list(WAREHOUSE_CENTER_SPAWN),
        person_specs=[],
    )
    config.ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE = False
    return config
