package com.example.qwen_android;

import static com.example.qwen_android.LLamaText.chatType;
import static com.example.qwen_android.LLamaText.peopleType;

import android.content.Context;
import android.graphics.Color;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.RecyclerView;


import java.util.ArrayList;
import java.util.List;

public class LLamaAdapter extends RecyclerView.Adapter<LLamaAdapter.LLamaHolder> {


    Context context;

    public void setContext(Context context) {
        this.context = context;
    }

    List<LLamaText> texts = new ArrayList<>();

    @NonNull
    @Override
    public LLamaHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_llama_txt, parent, false);
        LLamaHolder lLamaHolder = new LLamaHolder(view);
        return lLamaHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull LLamaHolder holder, int position) {
        LLamaText item = texts.get(position);

        holder.item_card.setCardBackgroundColor(Color.parseColor(item.type == peopleType ? "#3399ff" : "#E0E0E0"));
        holder.tv.setTextColor(Color.parseColor(item.type == peopleType ? "#FFFFFF" : "#333333"));

        switch (item.type) {
            case peopleType:
                holder.item_ll.setGravity(Gravity.RIGHT);
                break;
            case chatType:
            default:
                holder.item_ll.setGravity(Gravity.LEFT);
                break;
        }
        holder.tv.setText(  item.text);
    }

    public void upDate(List<LLamaText> texts) {
        this.texts.clear();
        if (texts != null) {
            this.texts.addAll(texts);
        }
        notifyDataSetChanged();
    }

    public void addDate(List<LLamaText> texts) {

        if (texts != null) {
            int pos = this.texts.size() - 1;
            this.texts.addAll(texts);
            notifyItemChanged(pos);
        }
    }

    @Override
    public int getItemCount() {
        return texts == null ? 0 : texts.size();
    }

    protected class LLamaHolder extends RecyclerView.ViewHolder {
        TextView tv;
        LinearLayout item_ll;
        CardView item_card;

        public LLamaHolder(View itemView) {
            super(itemView);
            tv = itemView.findViewById(R.id.text);
            item_card = itemView.findViewById(R.id.item_card);
            item_ll = itemView.findViewById(R.id.item_ll);

        }
    }
}
