from memswarm import InMemorySharedMemory, RedisSharedMemory

class LangSwarmIntegration:
    """
    A utility for integrating MemSwarm with LangSwarm workflows.
    """

    @staticmethod
    def setup_shared_memory(agent_wrappers, memory_type="in_memory", **kwargs):
        """
        Setup shared memory for a list of LangSwarm AgentWrapper instances.

        Parameters:
        - agent_wrappers: List of AgentWrapper instances.
        - memory_type: Type of shared memory ("in_memory" or "redis").
        - kwargs: Additional parameters for the memory implementation.
        """
        if memory_type == "in_memory":
            shared_memory = InMemorySharedMemory()
        elif memory_type == "redis":
            shared_memory = RedisSharedMemory(**kwargs)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

        for agent in agent_wrappers:
            agent.shared_memory = shared_memory
