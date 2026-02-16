ADD_COMMENT_MUTATION = """
mutation AddComment($input: AddCommentInput!) {
    addComment(input: $input) {
        clientMutationId
        commentEdge {
            node {
                id
                body
                createdAt
                author {
                    login
                }
            }
        }
        subject {
            ... on Issue {
                id
                title
                body
            }
            ... on PullRequest {
                id
                title
                body
            }
        }
    }
}
"""
