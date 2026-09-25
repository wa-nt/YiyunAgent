# Python asyncio 笔记

- `asyncio.gather` 并发跑多个协程，`return_exceptions=True` 避免一个失败拖垮全部
- 限制并发用 `asyncio.Semaphore`，不要在循环里裸 `create_task` 上千个协程
- fire-and-forget 任务要保存引用（`task.add_done_callback`），否则可能被 GC
- `asyncio.wait_for` 加超时；取消任务时记得处理 `CancelledError`
- 事件循环里禁止阻塞调用，阻塞 I/O 用 `asyncio.to_thread` 挪到线程池
