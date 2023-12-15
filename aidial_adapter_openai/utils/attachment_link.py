from pydantic import BaseModel


def to_absolute_url(dial_url: str, document_url_or_path: str) -> str:
    """
    Specification:
    - If URL starts with protocol "protocol://" then it's an URL.
    - If it starts with "/" then it's a relative path.
    - Otherwise, it's a corrupted link.
    - If it's a relative path, it needs to be "absolutized" first by concatenating "URL to core",
      "/v1/files" prefix for File API and "absolute" flag.

    Note: The dial does not follow RFC 1808 - Relative Uniform Resource Locators, so we cannot use urlparse here.
    """
    # Assume that "protocol://" means http or https protocol
    if document_url_or_path.startswith(
        "http://"
    ) or document_url_or_path.startswith("https://"):
        return document_url_or_path
    elif document_url_or_path.startswith("/"):
        return f"{dial_url}/v1/files{document_url_or_path}?path=absolute"
    else:
        raise ValueError(f"Corrupted link: {document_url_or_path}")


class AttachmentLink(BaseModel):
    """
    Link to the attached document in the Dial could be an URL or a relative path to the root of the Dial file API.
    This class represents both of them.

    Fields:
    - dial_link: The original URL or relative path. Should be to refer the attachment in Dial API or messages.
    - absolute_url: The absolute URL. Should be used to get the content of the attached document.
    """

    dial_link: str
    absolute_url: str

    def __str__(self) -> str:
        return self.dial_link

    @property
    def is_dial_relative_path(self) -> bool:
        return self.dial_link.startswith("/")

    @classmethod
    def from_url_or_path(
        cls, dial_url: str, url_or_path: str
    ) -> "AttachmentLink":
        return cls(
            dial_link=url_or_path,
            absolute_url=to_absolute_url(dial_url, url_or_path),
        )
