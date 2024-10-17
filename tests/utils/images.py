from aidial_adapter_openai.utils.resource import Resource

pic_1_1 = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC",
)

pic_2_2 = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGNk+M/AwMDAxMDAwMDAAAAMHgEDBINhkwAAAABJRU5ErkJggg==",
)

pic_3_3 = Resource.from_base64(
    type="image/png",
    data_base64="iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAIAAADZSiLoAAAAF0lEQVR4nGNkYPjPwMDAwMDAxAADCBYAG10BBdmz9y8AAAAASUVORK5CYII=",
)


def data_url(resource: Resource) -> str:
    return resource.to_data_url()
