import jax
jax.config.update("jax_default_matmul_precision", "highest")

from absl import app
from absl import flags

from ml_collections import config_flags

from src import train, evaluate

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

def main(argv):
    model, ckptdir = train.train(FLAGS.config)
    evaluate.eval(FLAGS.config, ckptdir, model)

if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)