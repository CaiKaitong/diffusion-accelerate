#!/usr/bin/env python3

import logging
import click

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MESSAGE = (
    "BAC4 no longer supports offline BAC schedule analysis. "
    "Use cache_mode='original' or cache_mode='skip' directly in the policy sampling loop."
)


def get_optimal_cache_update_steps(*args, **kwargs):
    raise RuntimeError(MESSAGE)


@click.command()
def main():
    logger.error(MESSAGE)
    raise SystemExit(1)


if __name__ == '__main__':
    main()
