from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from queue import Queue
from threading import Event

class AnalysisAgent:
    """Agent for orchestrating video analysis tasks"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        self.tasks: Queue[Callable] = Queue()
        self.results: Queue[Any] = Queue()
        self.stop_event = Event()
        
    def add_task(self, task: Callable, *args, **kwargs):
        """Add task to agent's queue"""
        self.tasks.put((task, args, kwargs))
        
    def get_result(self) -> Any:
        """Get next result from agent"""
        return self.results.get()
        
    def stop(self):
        """Signal agent to stop processing"""
        self.stop_event.set()

class AnalysisOrchestrator:
    """Orchestrates multiple analysis agents"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.agents: Dict[str, AnalysisAgent] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger("orchestrator")
        
    def create_agent(self, name: str) -> AnalysisAgent:
        """Create new analysis agent"""
        agent = AnalysisAgent(name)
        self.agents[name] = agent
        return agent
        
    def execute_parallel(self, tasks: List[Dict]) -> List[Dict]:
        """Execute tasks in parallel using agents"""
        futures = []
        results = []
        
        for task in tasks:
            agent_name = task.get('agent', 'default')
            if agent_name not in self.agents:
                self.create_agent(agent_name)
                
            agent = self.agents[agent_name]
            future = self.executor.submit(
                self._execute_agent_task,
                agent,
                task['function'],
                *task.get('args', []),
                **task.get('kwargs', {})
            )
            futures.append(future)
            
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task execution error: {e}")
                
        return results
        
    def execute_sequential(self, tasks: List[Dict]) -> List[Dict]:
        """Execute tasks sequentially"""
        results = []
        for task in tasks:
            try:
                agent_name = task.get('agent', 'default')
                if agent_name not in self.agents:
                    self.create_agent(agent_name)
                    
                agent = self.agents[agent_name]
                result = self._execute_agent_task(
                    agent,
                    task['function'],
                    *task.get('args', []),
                    **task.get('kwargs', {})
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task execution error: {e}")
                
        return results
        
    def _execute_agent_task(self, agent: AnalysisAgent, 
                           func: Callable, *args, **kwargs) -> Dict:
        """Execute single task with agent"""
        try:
            agent.add_task(func, *args, **kwargs)
            return {
                'agent': agent.name,
                'status': 'success',
                'result': func(*args, **kwargs)
            }
        except Exception as e:
            return {
                'agent': agent.name,
                'status': 'error',
                'error': str(e)
            }
            
    def shutdown(self):
        """Shutdown orchestrator and all agents"""
        for agent in self.agents.values():
            agent.stop()
        self.executor.shutdown()
