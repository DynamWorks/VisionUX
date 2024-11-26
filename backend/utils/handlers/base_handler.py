from .handler_interface import MessageHandler

class BaseMessageHandler(MessageHandler):
    """Base implementation of message handler"""
    
    async def handle(self, websocket, message):
        """Default implementation that should be overridden"""
        await self.send_error(websocket, "Handler not implemented")
