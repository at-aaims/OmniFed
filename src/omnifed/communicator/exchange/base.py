from abc import ABC, abstractmethod


class ModelExchange(ABC):
    """Exchange a complete model using implementation-owned chunk scheduling. 
    This interface lets each implementation own the entire chunk loop."""

    @abstractmethod
    def exchange_client_model(
        self,
        *,
        local_state,
        local_units,
        algorithm,
        round_id,
        global_step=None,
        timer=None,
    ):
        """Submit a client's model and receive the global update."""
        raise NotImplementedError

    @abstractmethod
    def aggregate_server_round(
        self,
        *,
        global_state,
        algorithm,
        round_id,
        timer=None,
    ):
        """Aggregate client models, publish updates, and update server state."""
        raise NotImplementedError