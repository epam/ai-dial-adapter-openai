from aidial_adapter_openai.configuration.app_config import Vendor
from aidial_adapter_openai.providers.alibaba import AlibabaAdapter
from aidial_adapter_openai.providers.vendor_adapter import (
    NoOpVendorAdapter,
    VendorAdapter,
)

_VENDOR_ADAPTERS: dict[Vendor, VendorAdapter] = {
    Vendor.ALIBABA: AlibabaAdapter(),
}

_NO_OP_ADAPTER = NoOpVendorAdapter()


def get_vendor_adapter(vendor: Vendor) -> VendorAdapter:
    return _VENDOR_ADAPTERS.get(vendor, _NO_OP_ADAPTER)
