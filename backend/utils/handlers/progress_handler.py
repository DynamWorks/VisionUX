import logging
import time
from typing import Any, Dict, Optional
from .base_handler import BaseHandler

class ProgressHandler(BaseHandler):
    """Handler for tracking operation progress"""
    
    def __init__(self):
        super().__init__("progress", "system")
        self.operations = {}
        self.logger.info("Initialized ProgressHandler")

    def _handle_impl(self, data: Any) -> Dict:
        """Handle progress update"""
        try:
            operation_id = data.get('operation_id')
            if not operation_id:
                raise ValueError("Missing operation_id")

            progress = data.get('progress', 0)
            status = data.get('status', 'in_progress')
            message = data.get('message', '')

            self._update_operation(operation_id, progress, status, message)

            return {
                'status': 'success',
                'operation_id': operation_id,
                'progress': self.get_operation_progress(operation_id),
                'timestamp': time.time()
            }
        except Exception as e:
            self._log_error(e, "Error updating progress")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def validate(self, data: Any) -> bool:
        """Validate progress update data"""
        if not isinstance(data, dict):
            return False
        if 'operation_id' not in data:
            return False
        if 'progress' in data and not isinstance(data['progress'], (int, float)):
            return False
        return True

    def start_operation(self, operation_id: str, operation_type: str, total_steps: int = 100) -> Dict:
        """Start tracking a new operation"""
        if operation_id in self.operations:
            self.logger.warning(f"Operation {operation_id} already exists")
            return self.get_operation_progress(operation_id)

        operation = {
            'id': operation_id,
            'type': operation_type,
            'start_time': time.time(),
            'update_time': time.time(),
            'end_time': None,
            'progress': 0,
            'total_steps': total_steps,
            'current_step': 0,
            'status': 'started',
            'message': 'Operation started',
            'error': None
        }
        self.operations[operation_id] = operation
        return operation

    def update_operation(self, operation_id: str, progress: float, message: str = '') -> Dict:
        """Update operation progress"""
        return self._update_operation(operation_id, progress, 'in_progress', message)

    def complete_operation(self, operation_id: str, message: str = 'Operation completed') -> Dict:
        """Mark operation as completed"""
        return self._update_operation(operation_id, 100, 'completed', message)

    def fail_operation(self, operation_id: str, error: str) -> Dict:
        """Mark operation as failed"""
        return self._update_operation(operation_id, -1, 'failed', error)

    def get_operation_progress(self, operation_id: str) -> Dict:
        """Get current progress of an operation"""
        if operation_id not in self.operations:
            raise ValueError(f"Operation {operation_id} not found")
        return self.operations[operation_id].copy()

    def _update_operation(self, operation_id: str, progress: float, status: str, message: str) -> Dict:
        """Internal method to update operation status"""
        if operation_id not in self.operations:
            raise ValueError(f"Operation {operation_id} not found")

        operation = self.operations[operation_id]
        operation['update_time'] = time.time()
        operation['progress'] = max(0, min(100, progress))
        operation['status'] = status
        operation['message'] = message

        if status in ['completed', 'failed']:
            operation['end_time'] = time.time()

        if status == 'failed':
            operation['error'] = message

        return operation.copy()

    def cleanup(self) -> None:
        """Cleanup handler resources"""
        super().cleanup()
        # Clean up old operations (keep only last 24 hours)
        current_time = time.time()
        cutoff_time = current_time - (24 * 60 * 60)
        self.operations = {
            op_id: op for op_id, op in self.operations.items()
            if op['update_time'] > cutoff_time
        }

    def get_active_operations(self) -> Dict[str, Dict]:
        """Get all active operations"""
        return {
            op_id: op.copy() for op_id, op in self.operations.items()
            if op['status'] not in ['completed', 'failed']
        }

    def get_operation_stats(self) -> Dict:
        """Get statistics about operations"""
        total = len(self.operations)
        active = len([op for op in self.operations.values() if op['status'] == 'in_progress'])
        completed = len([op for op in self.operations.values() if op['status'] == 'completed'])
        failed = len([op for op in self.operations.values() if op['status'] == 'failed'])

        return {
            'total': total,
            'active': active,
            'completed': completed,
            'failed': failed,
            'timestamp': time.time()
        }
