from openai.types.chat import ChatCompletionContentPartImageParam

from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.image import ImageResource


async def image_content_part(
    resource: Resource,
) -> ChatCompletionContentPartImageParam:
    image = await ImageResource.from_resource(resource, detail=None)
    return image.to_content_part()
