import os
import json
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class Post:
    def __init__(self, alias, url_image, is_video, multiple_image, tags, mentions, description, date, numberLikes):
        self.alias = alias
        self.url_image = url_image
        self.is_video = is_video
        self.multiple_image = multiple_image
        self.tags = tags
        self.mentions = mentions
        self.description = description
        self.date = date
        self.number_likes = numberLikes

    @classmethod
    def from_json(cls, data,alias):
        return cls(
            alias,
            data.get("urlImage"),
            data.get("isVideo"),
            data.get("multipleImage"),
            data.get("tags"),
            data.get("mentions"),
            data.get("description"),
            data.get("date"),
            data.get("numberLikes")
        )

    @staticmethod
    def load_posts_from_folder(folder_path):
        posts = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                with open(os.path.join(folder_path, filename), encoding='utf-8') as file:
                    data = json.load(file)
                alias=data.get('alias')
                for post_data in data.get("posts", []):
                    post = Post.from_json(post_data,alias)
                    posts.append(post)
        data = {
            "alias": [post.alias for post in posts],
            "urlImage": [post.url_image for post in posts],
            "isVideo": [post.is_video for post in posts],
            "multipleImage": [post.multiple_image for post in posts],
            "tags": [post.tags for post in posts],
            "mentions": [post.mentions for post in posts],
            "description": [post.description for post in posts],
            "date": [post.date for post in posts],
            "numberLikes": [post.number_likes for post in posts]
        }
        return pd.DataFrame(data)


class Profile:
    def __init__(self, alias, number_posts, number_followers, number_following, website):
        self.alias = alias
        self.number_posts = number_posts
        self.number_followers = number_followers
        self.number_following = number_following
        self.website = website

    @classmethod
    def from_json(cls, data):
        return cls(data.get("alias"),
                   data.get("numberPosts"),
                   data.get("numberFollowers"),
                   data.get("numberFollowing"),
                   data.get("website"))

    @staticmethod
    def load_profiles_from_folder(folder_path):
        profiles = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                with open(os.path.join(folder_path, filename), encoding='utf-8') as file:
                    data = json.load(file)
                    profile = Profile.from_json(data)
                    profiles.append(profile)
        data = {
            "alias": [profile.alias for profile in profiles],
            "numberPosts": [profile.number_posts for profile in profiles],
            "numberFollowers": [profile.number_followers for profile in profiles],
            "numberFollowing": [profile.number_following for profile in profiles],
            "website": [profile.website for profile in profiles]
        }
        return pd.DataFrame(data)



