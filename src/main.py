from importlib import import_module
import os

# for docker
# executed_directory = os.environ["EXECUTED_DIRECTORY"]

# for local
executed_directory = "test_20230715"

imported_module = import_module("src." + executed_directory)
imported_module.main()
