import asyncio
from typing import Dict,Any

# --- 异步缓存监听器 (优化后) ---
class ResponseListener:
    def __init__(self):
        self._pending_futures: Dict[str, asyncio.Future] = {}

    def register_future(self, message_id: str, future: asyncio.Future):
        """
        将一个 Future 与指定的 message_id 关联并注册。
        这是在发送请求之前调用的。
        """
        print("开始注册？",self._pending_futures)
        if message_id in self._pending_futures:
            raise ValueError(f"消息 ID {message_id} 已经被注册，不能重复等待。")
        self._pending_futures[message_id] = future
        print(f"[监听器] 注册 Future (ID: {message_id})。")

    def unregister_future(self, message_id: str):
        """
        从监听器中移除指定的 Future。
        在 Future 完成、超时或被取消后调用。
        """
        if message_id in self._pending_futures:
            del self._pending_futures[message_id]
            print(f"[监听器] 移除 Future (ID: {message_id})。")

    def deliver_message(self, message_data: Dict[str, Any]) -> bool:
        """
        将收到的消息投递给监听器。
        如果消息的 ID 匹配某个等待的 Future，则设置 Future 结果并返回 True。
        否则返回 False。
        """
        message_id = message_data.get("id")
        if message_id in self._pending_futures:
            future = self._pending_futures[message_id]
            if not future.done():
                future.set_result(message_data)
                print(f"[监听器] 成功投递并匹配到 ID: {message_id} 的响应。Future 已设置。")
                # 消息一旦被处理，就可以从 pending_futures 中移除
                # 但为了安全，可以在 register_for_response 的 finally 或调用端处理，
                # 这里只负责设置结果。
                # self.unregister_future(message_id) # 也可以在这里移除
                return True
            else:
                print(f"[监听器] 警告: ID: {message_id} 的 Future 已经完成/取消，但消息再次到达。")
                return False
        return False

    def cleanup(self):
        """清理所有未完成的 Future (例如，当连接关闭时)"""
        for msg_id, future in list(self._pending_futures.items()):
            if not future.done():
                future.cancel() # 取消未完成的 Future
                print(f"[监听器] 清理并取消了 ID: {msg_id} 的等待 Future。")
        self._pending_futures.clear()