import hydra

from containers import Trainer


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    trainer = Trainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()