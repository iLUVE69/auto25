#include <bits/stdc++.h>
using namespace std;
vector<int> flattenmatrix(vector<vector<int>> matrix){

    int n = matrix.size();
    int m = matrix[0].size();
    vector<int> ans;
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            ans.push_back(matrix[i][j]);
        }    
    }
    return ans;     
}

bool searchMatrix(vector<vector<int>>& matrix, int target) {
    vector<int> ans = flattenmatrix(matrix);
    int n = ans.size();
    int low = 0;
    int high = n-1;
    while (low<=high){
        int mid = (low+high)/2;
        if(ans[mid]==target){
            return true;
        }
        else if(ans[mid]>target){
            high = mid -1;

        }
        else{
            low = mid +1;
        }
    }return false;
}


int main(){

    vector<vector<int>> matrix = {{1,3,5,7},{10,11,16,20},{23,30,34,60}};
    int target = 37;
    bool ans = searchMatrix(matrix,target);
    cout<<ans<<endl;
    return 0;
}