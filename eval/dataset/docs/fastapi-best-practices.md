# FastAPI 最佳实践

记录时间：2026-09-14（上周）。

- 路由按业务模块拆分 APIRouter，不要全堆在 main.py
- 依赖注入（Depends）管理数据库会话，lifespan 里做初始化和清理
- Pydantic model 做请求/响应校验，response_model 显式声明
- 长时间任务丢后台（BackgroundTasks 或独立 worker），不要阻塞事件循环
- 异常统一用 HTTPException + exception handler，返回一致的错误结构
- 测试用 httpx.ASGITransport 直接打 ASGI app，不用起服务
