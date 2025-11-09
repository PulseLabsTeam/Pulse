Getting Started
================

See :doc:`GETTING_STARTED.md <getting_started>` for a comprehensive getting started guide.

Quick Example
-------------

.. code-block:: python

   import asyncio
   from pulseos import Runtime, Agent, SurvivalConstraint

   class MyAgent(Agent):
       def __init__(self, agent_id: str):
           super().__init__(agent_id)
           self.value = 0.0
           self.target = 0.8
       
       async def step(self) -> dict:
           error = self.target - self.value
           self.value += self.learning_rate * error
           return {"value": self.value}
       
       def get_performance_metric(self) -> float:
           error = abs(self.target - self.value)
           return 1.0 - error

   async def main():
       constraint = SurvivalConstraint(threshold=0.7)
       runtime = Runtime(constraint=constraint)
       
       agent = MyAgent("agent_1")
       runtime.register_agent("agent_1", agent)
       
       await runtime.run(max_steps=100)
       
       stats = runtime.get_statistics()
       print(f"Survival signal: {stats['average_survival_signal']:.3f}")

   asyncio.run(main())

