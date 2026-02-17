FULL_RESOLVE_ISSUE_QUERY = """
query FullResolve($id: ID!) {
    node(id: $id) {
        ... on Issue {
            id
            title
            repository {
                name
                owner {
                    ... on Organization {
                        projectsV2(first: 20) {
                            nodes {
                                id
                                title
                                fields(first: 50) {
                                    nodes {
                                        __typename
                                        ... on ProjectV2SingleSelectField {
                                            id
                                            name
                                            dataType
                                            options {
                                                id
                                                name
                                                color
                                            }
                                        }
                                        ... on ProjectV2IterationField {
                                            id
                                            name
                                            dataType
                                        }
                                        ... on ProjectV2FieldCommon {
                                            id
                                            name
                                            dataType
                                        }
                                    }
                                }
                                items(first: 50) {
                                    nodes {
                                        id
                                        content {
                                            ... on Issue {
                                                id
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""

UPDATE_ITEM_FIELD_MUTATION = """
mutation UpdateItemField($input: UpdateProjectV2ItemFieldValueInput!) {
    updateProjectV2ItemFieldValue(input: $input) {
        projectV2Item { id }
    }
}
"""

PROJECT_ISSUES_WITH_FIELDS_QUERY = """
query ProjectIssuesWithAllCustomFields(
    $owner: String!,
    $projectNumber: Int!,
    $after: String
) {
    organization(login: $owner) {
        projectV2(number: $projectNumber) {
            items(first: 50, after: $after) {
                pageInfo {
                    hasNextPage
                    endCursor
                }
                nodes {
                    fieldValues(first: 50) {
                        nodes {
                            ... on ProjectV2ItemFieldSingleSelectValue {
                                field {
                                    ... on ProjectV2FieldCommon {
                                        name
                                        dataType
                                    }
                                }
                                name
                            }
                            ... on ProjectV2ItemFieldIterationValue {
                                field {
                                    ... on ProjectV2FieldCommon {
                                        name
                                        dataType
                                        id
                                    }
                                }
                                iterationId
                                title
                                startDate
                                duration
                            }
                        }
                    }
                    content {
                        ... on Issue {
                            number
                            title
                            body
                            state
                            createdAt
                            comments {
                                totalCount
                            }
                            author {
                                login
                            }
                            authorAssociation
                            labels(first: 50) {
                                nodes { name }
                            }
                            assignees(first: 50) {
                                nodes { login }
                            }
                            repository {
                                nameWithOwner
                                primaryLanguage {
                                    name
                                }
                                diskUsage
                            }
                        }
                    }
                }
            }
        }
    }
}
"""

ISSUE_PROJECTS_BY_NODE_QUERY = """
query IssueProjectsByNode($issueId: ID!) {
    node(id: $issueId) {
        ... on Issue {
            projectItems(first: 20) {
                nodes {
                    createdAt
                    project {
                        id
                        number
                    }
                }
            }
        }
    }
}
"""

PROJECT_ITERATION_DURATION_QUERY = """
query ProjectIterationDuration($projectId: ID!) {
    node(id: $projectId) {
        ... on ProjectV2 {
            fields(first: 20) {
                nodes {
                    ... on ProjectV2IterationField {
                        id
                        name
                        configuration {
                            iterations {
                                id
                                title
                                startDate
                                duration
                            }
                        }
                    }
                }
            }
        }
    }
}
"""

PROJECT_ITERATION_FIELD_QUERY = """
query ProjectIterationField($projectId: ID!) {
    node(id: $projectId) {
        ... on ProjectV2 {
            fields(first: 50) {
                nodes {
                    ... on ProjectV2IterationField {
                        id
                        name
                        configuration {
                            iterations {
                                id
                                title
                                startDate
                                duration
                            }
                        }
                    }
                    ... on ProjectV2FieldCommon {
                        id
                        name
                        dataType
                    }
                }
            }
        }
    }
}
"""

UPDATE_ISSUE_SPRINT_MUTATION = """
mutation UpdateIssueSprint($projectId: ID!, $itemId: ID!, $fieldId: ID!, $iterationId: String!) {
    updateProjectV2ItemFieldValue(input: { projectId: $projectId itemId: $itemId fieldId: $fieldId value: { iterationId: $iterationId } }) {
        projectV2Item { id }
    }
}
"""
