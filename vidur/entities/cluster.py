import json
import math

from vidur.config import BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig
from vidur.entities.base_entity import BaseEntity
from vidur.entities.replica import Replica
from vidur.logger import init_logger

logger = init_logger(__name__)


class Cluster(BaseEntity):
    def __init__(
        self,
        cluster_config: ClusterConfig,
        metrics_config: MetricsConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Cluster.generate_id()
        self._config = cluster_config
        self._generator_config = generator_config

        # get metrics config
        self._output_dir = metrics_config.output_dir

        # Init replica object handles
        self._replicas = {}

        for _ in range(self._config.num_replicas):
            replica = Replica(self._config.replica_config, generator_config)
            self._replicas[replica.id] = replica

        if metrics_config.write_json_trace:
            self._write_cluster_info_to_file()

    @property
    def replicas(self):
        return self._replicas

    @property
    def num_replicas(self) -> int:
        return len(self._replicas)

    @property
    def cost_per_hour(self) -> float:
        num_devices = self.num_replicas * self._config.replica_config.world_size
        num_nodes = math.ceil(num_devices / self._config.replica_config.node_config.num_devices_per_node)
        return self._config.replica_config.node_config.cost_per_hour * num_nodes

    def add_replica(self) -> Replica:
        replica = Replica(self._config.replica_config, self._generator_config)
        self._replicas[replica.id] = replica
        return replica

    def free_replica_with_id(self, replica_id: int) -> None:
        del self._replicas[replica_id]

    def free_replica(self) -> None:
        """
        Free a replica from the cluster
        Note: This is meant to be used only by the autograded code
        """
        replica_id = next(iter(self._replicas))
        self.free_replica_with_id(replica_id)

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_replicas": len(self._replicas),
        }

    def _write_cluster_info_to_file(self) -> None:
        replica_dicts = [replica.to_dict() for replica in self._replicas.values()]
        cluster_info = {"replicas": replica_dicts}

        cluster_file = f"{self._output_dir}/cluster.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_info, f)
