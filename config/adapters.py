import uuid
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter

class SocialAccountAdapter(DefaultSocialAccountAdapter):
    def populate_user(self, request, sociallogin, data):
        user = super().populate_user(request, sociallogin, data)
        # username 自動生成，避免要求使用者填
        if not user.username:
            user.username = f"u_{uuid.uuid4().hex[:10]}"
        return user
