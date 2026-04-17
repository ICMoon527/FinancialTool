# -*- coding: utf-8 -*-
"""
API 响应格式测试脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.v1.schemas.common import (
    ApiResponse,
    ApiError,
    ApiResponseMeta,
    success_response,
    error_response
)


def test_success_response():
    """测试成功响应"""
    print("=" * 80)
    print("测试 1: 标准成功响应")
    print("=" * 80)
    
    # 测试 success_response() 工具函数
    data = {"stocks": ["600519", "000001"], "count": 2}
    resp = success_response(data=data, message="获取股票列表成功")
    
    print("\n1.1 success_response() 工具函数:")
    print(f"   状态码: {resp.code}")
    print(f"   消息: {resp.message}")
    print(f"   数据: {resp.data}")
    print(f"   元数据: {resp.meta.model_dump()}")
    print(f"   JSON: {resp.model_dump_json(indent=2)}")
    
    # 测试 ApiResponse 类直接使用
    print("\n1.2 ApiResponse 类直接使用:")
    api_resp = ApiResponse(
        code=201,
        message="分析已创建",
        data={"analysis_id": "uuid-1234"}
    )
    print(f"   JSON: {api_resp.model_dump_json(indent=2)}")
    
    print("\n✅ 成功响应格式测试通过!")


def test_error_response():
    """测试错误响应"""
    print("\n" + "=" * 80)
    print("测试 2: 标准错误响应")
    print("=" * 80)
    
    # 测试 error_response() 工具函数
    print("\n2.1 error_response() 工具函数:")
    err_resp = error_response(
        error="validation_error",
        message="股票代码格式错误",
        code=400,
        detail={"field": "stock_code", "value": "invalid"}
    )
    print(f"   状态码: {err_resp.code}")
    print(f"   错误类型: {err_resp.error}")
    print(f"   消息: {err_resp.message}")
    print(f"   详情: {err_resp.detail}")
    print(f"   JSON: {err_resp.model_dump_json(indent=2)}")
    
    # 测试不同的错误类型
    print("\n2.2 不同错误类型示例:")
    
    error_types = [
        ("not_found", 404, "资源不存在"),
        ("unauthorized", 401, "未授权访问"),
        ("forbidden", 403, "禁止访问"),
        ("internal_error", 500, "服务器内部错误"),
        ("rate_limit_exceeded", 429, "请求频率超限"),
    ]
    
    for error_type, code, message in error_types:
        err = error_response(error=error_type, message=message, code=code)
        print(f"   {code} - {error_type}: {message}")
    
    print("\n✅ 错误响应格式测试通过!")


def test_meta_data():
    """测试元数据"""
    print("\n" + "=" * 80)
    print("测试 3: 响应元数据")
    print("=" * 80)
    
    # 测试 ApiResponseMeta
    print("\n3.1 ApiResponseMeta:")
    meta = ApiResponseMeta(request_id="test-request-id-1234")
    print(f"   时间戳: {meta.timestamp}")
    print(f"   请求 ID: {meta.request_id}")
    print(f"   API 版本: {meta.api_version}")
    print(f"   JSON: {meta.model_dump_json(indent=2)}")
    
    # 测试自定义 request_id
    print("\n3.2 自定义 request_id:")
    custom_resp = success_response(data=None, request_id="custom-id-5678")
    print(f"   Request ID: {custom_resp.meta.request_id}")
    
    # 测试自动生成 request_id
    print("\n3.3 自动生成 request_id:")
    auto_resp = error_response(error="test", message="test", code=400)
    print(f"   自动生成的 Request ID: {auto_resp.meta.request_id}")
    
    print("\n✅ 元数据测试通过!")


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "=" * 80)
    print("测试 4: 向后兼容性")
    print("=" * 80)
    
    from api.v1.schemas.common import ErrorResponse, SuccessResponse
    
    # 测试旧格式的 ErrorResponse 仍然可用
    print("\n4.1 旧格式 ErrorResponse:")
    old_error = ErrorResponse(
        error="old_format",
        message="旧格式仍然支持"
    )
    print(f"   JSON: {old_error.model_dump_json(indent=2)}")
    
    # 测试旧格式的 SuccessResponse 仍然可用
    print("\n4.2 旧格式 SuccessResponse:")
    old_success = SuccessResponse(
        success=True,
        message="操作成功",
        data={"key": "value"}
    )
    print(f"   JSON: {old_success.model_dump_json(indent=2)}")
    
    print("\n✅ 向后兼容性测试通过!")


def main():
    """主函数"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "API 响应格式测试" + " " * 42 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        test_success_response()
        test_error_response()
        test_meta_data()
        test_backward_compatibility()
        
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "✅ 所有测试通过!" + " " * 40 + "║")
        print("╚" + "=" * 78 + "╝")
        
        print("\n📝 API 响应格式说明:")
        print("   - 成功响应: success_response(data, message, code, request_id)")
        print("   - 错误响应: error_response(error, message, code, detail, request_id)")
        print("   - 标准格式包含: code, message, data/error, meta(timestamp, request_id, api_version)")
        print("   - 保持向后兼容，原有的 ErrorResponse 和 SuccessResponse 仍然可用")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
