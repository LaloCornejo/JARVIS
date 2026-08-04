pub mod run_command;
pub mod execute_python;
pub mod web_search;
pub mod fetch_url;
pub mod time;
pub mod files;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time() {
        let result = time::execute("UTC");
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.exit_code, Some(0));
        assert!(!res.stdout.is_empty());
    }

    #[tokio::test]
    async fn test_run_command_allowed() {
        let result = run_command::execute("whoami", 5).await;
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.exit_code, Some(0));
        assert!(!res.stdout.is_empty());
    }

    #[tokio::test]
    async fn test_run_command_blocked() {
        let result = run_command::execute("rm -rf /", 5).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_python_safe() {
        let code = "print('hello world')";
        let result = execute_python::execute(code, 5).await;
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.exit_code, Some(0));
        assert!(res.stdout.contains("hello world"));
    }

    #[tokio::test]
    async fn test_execute_python_blocked() {
        let code = "import os; os.system('ls')";
        let result = execute_python::execute(code, 5).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_web_search() {
        let result = web_search::execute("rust", 1).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_read_file() {
        let result = files::read_file("nonexistent.txt", 10).await;
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.exit_code, Some(1));
        assert!(res.stderr.contains("does not exist"));
    }

    #[tokio::test]
    async fn test_write_file() {
        let result = files::write_file("test.txt", "hello", false).await;
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.exit_code, Some(0));
        // Clean up
        tokio::fs::remove_file("test.txt").await.unwrap();
    }

    #[tokio::test]
    async fn test_list_directory() {
        let result = files::list_directory(".", None).await;
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.exit_code, Some(0));
    }

    #[tokio::test]
    async fn test_file_info() {
        let result = files::file_info("Cargo.toml").await;
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.exit_code, Some(0));
        assert!(res.stdout.contains("file"));
    }
}