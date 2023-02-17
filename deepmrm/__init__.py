from pathlib import Path
import yaml
import os
import dotenv

dotenv.load_dotenv()

# useful global variables for project
package_name = Path(__file__).parents[0].stem
project_dir = Path(__file__).parents[1]

private_project_dir = Path(os.getenv('PRIVATE_REPO_DIR'))
if private_project_dir.exists():
    data_dir = private_project_dir / 'data'
    reports_dir = private_project_dir / 'reports'
else:
    data_dir = project_dir / 'data'
    reports_dir = project_dir / 'reports'

model_dir = project_dir / 'models'

def get_yaml_config():
    yaml_config = yaml.load(
                        open(project_dir/'config.yml', 'r'), 
                        Loader=yaml.FullLoader)

    return yaml_config

# def load_dotenv(env_fname='.env'):
#     dotenv.load_dotenv(project_dir / env_fname)

__all__ = (
    "project_dir",
    "data_dir",
    "model_dir",
    "reports_dir",
    "get_yaml_config",
    "load_dotenv",
)