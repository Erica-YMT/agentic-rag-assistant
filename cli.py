from app.agent.agent import Agent

def run_cli():
    # ==========================
    # 创建 Agent
    # ==========================
    agent = Agent()
    # 当前命令行会话 ID
    session_id = "cli"
    print("===================")
    print("AI Agent启动")
    print("输入 e(exit) 或 q(quit) 退出")
    print("===================")

    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n程序已退出")
            break
        if question.lower() in {
            "e",
            "q"
        }:
            print("程序已退出")
            break
        if not question:
            continue
        try:
            answer = agent.run(
                session_id,
                question
            )
        except Exception as exc:
            print("\nAgent运行失败:")
            print(exc)
            continue
        print("\nAI:")
        print(answer)
if __name__ == "__main__":
    run_cli()