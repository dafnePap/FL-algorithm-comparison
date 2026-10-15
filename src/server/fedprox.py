from base import ServerBase
from client.fedprox import FedProxClient
from config.util import get_args

# Initialize the FedProx server and client trainer
class FedProxServer(ServerBase):
    def __init__(self):
        super(FedProxServer, self).__init__(get_args(), "FedProx")

        self.trainer = FedProxClient(
            backbone=self.backbone(self.args.dataset),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_steps=self.args.local_steps,     
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )
        # Set the FedProx proximal term
        self.trainer.mu = self.args.mu

if __name__ == "__main__":
    server = FedProxServer()
    server.run()
