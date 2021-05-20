---
title: Azure笔记
date: 2021-05-19 19:56:39
tags: [Microsoft, Azure]
categories: 学习笔记
top_img:
cover: https://rebornas.blob.core.windows.net/rebornhome/Azure.jpg
---

{% meting "523250334" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# Create access token
1. Create a new `App registration` in `Azure Active Directory`.
2. Create a new `Client Secret` in `Certificates & secrets` tab of the `App registration` created in step 1.
3. Use the following code to generate the access token.

```python
    import adal

    # Your Subscription ID
    subscription_id = 'XXXXXXXXXXXXXXXXXX'
    # Tenant ID for your Azure Subscription
    TENANT_ID       = 'XXXXXXXXXXXXXXXXXX'
    # Your Service Principal CLIENT_ID
    CLIENT_ID       = 'XXXXXXXXXXXXXXXXXX'
    # Your Service Principal CLIENT_SECRET created in step 2
    CLIENT_SECRET   = 'XXXXXXXXXXXXXXXXXX'

    authority_url = 'https://login.microsoftonline.com/{}'.format(TENANT_ID)
    context = adal.AuthenticationContext(authority_url)
    token = context.acquire_token_with_client_credentials(
        resource='https://management.azure.com/',
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    print(token["accessToken"])
```