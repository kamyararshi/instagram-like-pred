# 巩昕锐
# 开发时间：2023/6/8 18:09
from DataFrame import Profile, Post
import pandas as pd
pd.set_option('display.width', 180)


def add_average(profiles, posts):
    # Group posts_df by alias and calculate the average numberLikes
    average_likes = posts.groupby('alias')['numberLikes'].mean().round()
    profiles['average_likes'] = profiles['alias'].map(average_likes)
    return profiles


def filtering(profiles, posts, number, average):
    # filtering
    profiles_filtered = profiles[(profiles['numberFollowers'] < number) & (profiles['average_likes'] < average)]
    posts_filtered = posts[posts['alias'].isin(profiles_filtered['alias'])]
    return profiles_filtered, posts_filtered


def categorize_website(website):
    if pd.isnull(website):
        return 'None'
    elif 'youtube' in website.lower():
        return 'Youtube'
    elif 'facebook' in website.lower():
        return 'Facebook'
    elif 'twitter' in website.lower():
        return 'Twitter'
    elif 'blog' in website.lower():
        return 'Blog'
    elif 'music' in website.lower() or 'spotify' in website.lower():
        return 'Music'
    else:
        return 'Other'


def change_date2weekday(posts):
    # change the dates of the posts to weekdays
    posts = posts.copy()
    posts.loc[:, 'date'] = pd.to_datetime(posts['date'])
    posts.loc[:, 'weekday_number'] = posts['date'].dt.weekday + 1
    posts_weekday = posts.drop('date', axis=1)
    return posts_weekday


def number2category(posts):
    # Problem??

    # Categorizing into 10 equally sized groups based on numberLikes
    # Category 10 are the 10% of posts with the highest likes
    # Category 1 are the 10% with the lowest likes
    posts = posts.sort_values('numberLikes', ascending=False)  # Sort the DataFrame by numberLikes in descending order
    quantile = pd.qcut(posts['numberLikes'], q=10, labels=False,
                       duplicates='drop')  # Calculate the quantiles for the groups
    posts['numberLikesCategory'] = quantile + 1  # Add 1 to make the group numbers start from 1 instead of 0
    return posts


def delete_Video_and_multipleImage(profiles, posts):
    # delete the Videos and posts with multiple images
    posts_filtered = posts[~(posts['isVideo'] | posts['multipleImage'])]
    profiles_filtering = profiles[profiles['alias'].isin(posts_filtered['alias'])]
    return profiles_filtering, posts_filtered


folder_path = "profiles"
profiles_df = Profile.load_profiles_from_folder(folder_path)
posts_df = Post.load_posts_from_folder(folder_path)
# Add the average_likes column to profiles_df
profiles_df_ave = add_average(profiles_df, posts_df)
# filtering, such that numberFollowers < 1.000.000 & average_likes  < 200.000
profiles_df_filter, posts_df_filter = filtering(profiles_df_ave, posts_df, number=1000000, average=200000)
# print(profiles_df_filter.shape)
# print(posts_df.shape, posts_df_filter.shape)
# delete posts with videos and multiple images
profiles_df_filter, posts_df_filter = delete_Video_and_multipleImage(profiles_df_filter, posts_df_filter)
# print(posts_df_filter.shape)
#
# max_num = posts_df_filter['numberLikes'].max()
# print(max_num)

# profiles_df['website_category'] = profiles_df['website'].apply(categorize_website)
# print(profiles_df['website_category'].value_counts())

# Storing day of the week and delete the column of date
posts_df_filter_weekday = change_date2weekday(posts_df_filter)
# Categorizing into 10 equally sized groups based on numberLikes
posts_df_filter_weekday_ = number2category(posts_df_filter_weekday)
# print(posts_df_filter_weekday[:5])
# print(posts_df_filter_weekday.shape)
